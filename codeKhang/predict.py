import argparse
from contextlib import nullcontext
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from PIL import Image, ImageDraw, ImageFont

from src.config.model_kwargs import build_model_kwargs
from src.data.transforms import ImageTransform, TextTransform
from src.models.multimodal_yolo import MultimodalYOLO
from src.rag.integration.runtime import prepare_image_prompt
from src.rag.retrieval.request_builder import build_retrieval_request
from src.rag.runtime import build_runtime_service
from src.rag.runtime.predict_bridge import (
    build_image_runtime_payload,
    extract_disease_hints_from_detection,
)
from src.rag.runtime.types import ImageRuntimeResult

try:
    import ultralytics.utils.ops as yolo_ops
    try:
        from ultralytics.utils.nms import non_max_suppression as ultralytics_nms
    except ImportError:
        ultralytics_nms = yolo_ops.non_max_suppression
except Exception:
    ultralytics_nms = None

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    Standalone NMS using torchvision.
    Expected prediction shape: [batch_size, 4 + num_classes, num_anchors]
    """
    device = prediction.device
    out = [torch.zeros((0, 6), device=device)] * prediction.shape[0]
    for i, x in enumerate(prediction):
        x = x.transpose(0, 1)  # [num_anchors, 4 + num_classes]
        box = x[:, :4]  # cx, cy, w, h
        cls_scores = x[:, 4:]

        conf, j = cls_scores.max(1, keepdim=True)
        x_cat = torch.cat((box, conf, j.float()), 1)
        x_cat = x_cat[conf.view(-1) > conf_thres]

        if not x_cat.shape[0]:
            continue

        box = x_cat[:, :4].clone()
        box[:, 0] = x_cat[:, 0] - x_cat[:, 2] / 2
        box[:, 1] = x_cat[:, 1] - x_cat[:, 3] / 2
        box[:, 2] = x_cat[:, 0] + x_cat[:, 2] / 2
        box[:, 3] = x_cat[:, 1] + x_cat[:, 3] / 2
        x_cat[:, :4] = box

        i_nms = torchvision.ops.nms(x_cat[:, :4], x_cat[:, 4], iou_thres)
        out[i] = x_cat[i_nms]

    return out

def prediction_score_summary(prediction: torch.Tensor, topk: int = 10) -> dict:
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]

    if prediction.ndim != 3 or prediction.shape[1] <= 4:
        return {
            "shape": tuple(prediction.shape),
            "num_candidates": 0,
            "max_conf": 0.0,
            "mean_conf": 0.0,
            "top_conf": [],
            "top_cls": [],
        }

    cls_scores = prediction[:, 4:, :].detach().float()
    conf, cls = cls_scores.max(dim=1)
    flat_conf = conf.reshape(-1)
    flat_cls = cls.reshape(-1)
    k = min(topk, flat_conf.numel())

    if k == 0:
        top_conf = torch.zeros(0, device=prediction.device)
        top_indices = torch.zeros(0, dtype=torch.long, device=prediction.device)
    else:
        top_conf, top_indices = torch.topk(flat_conf, k=k)

    top_cls = flat_cls[top_indices] if k else torch.zeros(0, dtype=torch.long)

    return {
        "shape": tuple(prediction.shape),
        "num_candidates": int(flat_conf.numel()),
        "max_conf": float(flat_conf.max().item()) if flat_conf.numel() else 0.0,
        "mean_conf": float(flat_conf.mean().item()) if flat_conf.numel() else 0.0,
        "top_conf": [float(value) for value in top_conf.detach().cpu()],
        "top_cls": [int(value) for value in top_cls.detach().cpu()],
    }

def run_nms(prediction: torch.Tensor, conf_thres: float, iou_thres: float, backend: str = "auto"):
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]

    selected_backend = backend
    if backend == "auto":
        selected_backend = "ultralytics" if ultralytics_nms is not None else "torchvision"

    if selected_backend == "ultralytics":
        if ultralytics_nms is None:
            raise RuntimeError("Ultralytics NMS is not available in this environment.")
        return ultralytics_nms(prediction, conf_thres=conf_thres, iou_thres=iou_thres, max_det=300), selected_backend

    if selected_backend == "torchvision":
        return non_max_suppression(prediction, conf_thres=conf_thres, iou_thres=iou_thres), selected_backend

    raise ValueError(f"Unsupported NMS backend: {backend}")

def load_class_names(config: dict, data_yaml_path: str | None) -> list:
    if data_yaml_path:
        candidate = Path(data_yaml_path)
    else:
        data_cfg = config.get("data", {})
        train_dir = data_cfg.get("train_image_dir") or data_cfg.get("val_image_dir")
        candidate = Path(train_dir) if train_dir else None

    if candidate is None:
        return []

    if candidate.is_dir():
        data_yaml = candidate / "data.yaml"
        if not data_yaml.exists():
            data_yaml = candidate.parent.parent / "data.yaml"
    else:
        data_yaml = candidate

    if not data_yaml.exists():
        return []

    with open(data_yaml, "r", encoding="utf-8") as f:
        data_yaml_cfg = yaml.safe_load(f) or {}

    names = data_yaml_cfg.get("names", [])
    if isinstance(names, list):
        return names
    if isinstance(names, dict):
        try:
            return [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
        except (TypeError, ValueError):
            return [name for _, name in sorted(names.items(), key=lambda item: str(item[0]))]
    return []


def load_checkpoint_state(model, checkpoint, allow_partial_load: bool = False):
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)

    if missing_keys:
        print(f"[!] Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"[!] Unexpected keys: {len(unexpected_keys)}")

    if (missing_keys or unexpected_keys) and not allow_partial_load:
        missing_preview = ", ".join(missing_keys[:5]) or "none"
        unexpected_preview = ", ".join(unexpected_keys[:5]) or "none"
        raise RuntimeError(
            "Checkpoint/model mismatch detected while loading weights.\n"
            f"Missing keys preview: {missing_preview}\n"
            f"Unexpected keys preview: {unexpected_preview}\n"
            "Predict should use the same config surface as training. "
            "If you intentionally want partial loading, rerun with --allow-partial-load."
        )

    return load_result


def get_color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (255, 65, 54),
        (46, 204, 64),
        (0, 116, 217),
        (255, 133, 27),
        (177, 13, 201),
        (127, 219, 255),
        (255, 220, 0),
        (1, 255, 112),
        (170, 170, 170),
        (240, 18, 190),
    ]

    if class_id < len(palette):
        return palette[class_id]

    r = (37 * class_id) % 255
    g = (17 * class_id) % 255
    b = (29 * class_id) % 255
    return (r, g, b)


def select_prompt_for_text_transform(prepared_prompt) -> str:
    return prepared_prompt.normalized_prompt


class _PreparedImageRuntimeServiceAdapter:
    def __init__(self, runtime_service, prepared_input):
        self._runtime_service = runtime_service
        self._prepared_input = prepared_input

    def prepare_image_runtime_result(
        self,
        raw_text: str,
        disease_hints: tuple[str, ...] = (),
    ) -> ImageRuntimeResult:
        planned = self._runtime_service.planner.plan(self._prepared_input)
        retrieval_request = build_retrieval_request(
            planned,
            disease_hints=disease_hints,
        )
        normalized_disease_hints = tuple(
            item.value
            for item in retrieval_request.filters
            if item.field == "disease_name"
        )
        return ImageRuntimeResult(
            prepared=self._prepared_input,
            planned=planned,
            retrieval_request=retrieval_request,
            disease_hints=normalized_disease_hints,
        )


def build_post_detection_rag_runtime_payload(
    runtime_service,
    raw_text: str,
    det,
    class_names: list[str],
):
    detection_rows = []
    if det is not None and len(det):
        for *box, conf, cls in det.detach().cpu():
            detection_rows.append(
                {
                    "class_id": int(cls),
                    "confidence": float(conf),
                }
            )

    disease_hints = extract_disease_hints_from_detection(
        detections=detection_rows,
        class_names=class_names,
        max_hints=3,
    )
    image_runtime_result = runtime_service.prepare_image_runtime_result(
        raw_text=raw_text,
        disease_hints=disease_hints,
    )
    return build_image_runtime_payload(image_runtime_result)


def build_image_rag_runtime_seam(
    raw_prompt: str,
    runtime_service,
    det,
    class_names: list[str],
    preprocessor=None,
    prepared_prompt=None,
):
    selected_preprocessor = preprocessor or runtime_service.preprocessor
    resolved_prompt = prepared_prompt or prepare_image_prompt(
        raw_prompt,
        preprocessor=selected_preprocessor,
    )
    rag_runtime_payload = None
    if det is not None:
        payload_runtime_service = runtime_service
        if resolved_prompt.prepared is not None:
            payload_runtime_service = _PreparedImageRuntimeServiceAdapter(
                runtime_service=runtime_service,
                prepared_input=resolved_prompt.prepared,
            )
        rag_runtime_payload = build_post_detection_rag_runtime_payload(
            runtime_service=payload_runtime_service,
            raw_text=raw_prompt,
            det=det,
            class_names=class_names,
        )
    return resolved_prompt, rag_runtime_payload


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_kwargs = build_model_kwargs(config)

    # 2. Khởi tạo Model
    print(
        "[*] Model config: "
        f"text_pooling={model_kwargs['text_pooling_mode']}, "
        f"fusion={model_kwargs['fusion_mode']}, "
        f"feature_source={model_kwargs['visual_feature_source']}, "
        f"baseline={model_kwargs['visual_baseline_mode']}, "
        f"dropout={model_kwargs['dropout_prob']}"
    )
    model = MultimodalYOLO(**model_kwargs)

    # => Quan trọng: Ép TẤT CẢ các module trong Model vào chung 1 Device
    model = model.to(device)

    # Tải trọng số
    print(f"[*] Đang nạp trọng số từ {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    load_checkpoint_state(model, checkpoint, allow_partial_load=args.allow_partial_load)
    model.eval()

    img_transform = ImageTransform(image_size=config["data"]["image_size"])
    text_transform = TextTransform(
        model_name=config["model"]["text_encoder"]["model_name"],
        max_length=config["model"]["text_encoder"]["max_text_length"],
    )
    class_names = load_class_names(config, args.data_yaml)

    print("[*] Preparing image and text prompt...")
    original_image = Image.open(args.image).convert("RGB")
    boxed_image = original_image.copy()
    image_tensor, img_meta = img_transform(original_image, return_meta=True)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    runtime_service = None
    rag_runtime_mode = "disabled"

    try:
        runtime_service = build_runtime_service()
        preprocessor = runtime_service.preprocessor
        normalization_mode = "gemini"
        rag_runtime_mode = "gemini_runtime"
    except Exception as exc:
        print(f"[!] Full RAG runtime unavailable: {exc}")
        from src.rag.integration.runtime import build_fallback_image_preprocessor

        preprocessor = build_fallback_image_preprocessor()
        normalization_mode = "fallback"
        rag_runtime_mode = "fallback_only"

    prepared_prompt = None
    rag_runtime_payload = None

    if runtime_service is not None:
        prepared_prompt, _ = build_image_rag_runtime_seam(
            raw_prompt=args.prompt,
            runtime_service=runtime_service,
            det=None,
            class_names=class_names,
            preprocessor=preprocessor,
        )
    else:
        prepared_prompt = prepare_image_prompt(args.prompt, preprocessor=preprocessor)

    print(f"[*] Normalization mode: {normalization_mode}")
    print(f"[*] Raw prompt: {prepared_prompt.raw_prompt}")
    print(f"[*] Normalized prompt: {prepared_prompt.normalized_prompt}")
    print(f"[*] Query type: {prepared_prompt.prepared.normalization.query_type.value}")
    print(
        "[*] Normalization notes: "
        f"{prepared_prompt.prepared.normalization.normalization_notes}"
    )

    text_encoded = text_transform(select_prompt_for_text_transform(prepared_prompt))
    input_ids = text_encoded["input_ids"].unsqueeze(0).to(device)
    attention_mask = text_encoded["attention_mask"].unsqueeze(0).to(device)

    print("[*] Running inference...")
    autocast_context = (
        torch.amp.autocast(device_type="cuda")
        if torch.cuda.is_available()
        else nullcontext()
    )
    with torch.inference_mode():
        with autocast_context:
            preds, attn_maps = model(image_tensor, input_ids, attention_mask, return_attn=True)
            
            # Khử nhiễu Ultralytics Detect trả về tuple 3 phần tử khi inference
            if isinstance(preds, tuple) or isinstance(preds, list):
                preds = preds[0] 
                
            score_summary = prediction_score_summary(preds, topk=args.debug_topk)
            nms_preds, nms_backend = run_nms(
                preds,
                conf_thres=args.conf,
                iou_thres=args.iou,
                backend=args.nms_backend,
            )
            det = nms_preds[0]

    if runtime_service is not None:
        _, rag_runtime_payload = build_image_rag_runtime_seam(
            raw_prompt=args.prompt,
            runtime_service=runtime_service,
            det=det,
            class_names=class_names,
            preprocessor=preprocessor,
            prepared_prompt=prepared_prompt,
        )

    # 5. In kết quả console
    print(f"\n==================================================")
    print(f"  KẾT QUẢ DỰ ĐOÁN PROMPT: '{prepared_prompt.normalized_prompt}'")
    print(f"==================================================")
    print(f"  NMS backend: {nms_backend}")
    print(f"  Raw prediction shape: {score_summary['shape']}")
    print(
        f"  Raw candidate scores: max={score_summary['max_conf']:.4f}, "
        f"mean={score_summary['mean_conf']:.4f}, "
        f"top{len(score_summary['top_conf'])}={score_summary['top_conf']}"
    )
    print(f"  Raw top class ids: {score_summary['top_cls']}")
    print(f"[*] RAG runtime mode: {rag_runtime_mode}")
    if rag_runtime_payload is not None:
        print(f"[*] RAG normalized query: {rag_runtime_payload['normalized_query']}")
        print(f"[*] RAG intent: {rag_runtime_payload['intent']}")
        print(f"[*] RAG disease hints: {rag_runtime_payload['disease_hints']}")
        print(f"[*] RAG query terms: {rag_runtime_payload['query_terms']}")
        print(f"[*] RAG topic hints: {rag_runtime_payload['topic_hints']}")
        print(f"[*] RAG filters: {rag_runtime_payload['filters']}")
    
    pad_w, pad_h = img_meta["pad"]
    scale = img_meta["scale"]
    orig_w, orig_h = img_meta["orig_size"]
    min_dim = min(orig_w, orig_h)
    line_width = max(2, int(min_dim * 0.006))
    font_size = max(14, int(min_dim * 0.035))
    font_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]

    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except OSError:
            continue

    if font is None:
        print("[!] Warning: TrueType font not found. Falling back to default font.")
        font = ImageFont.load_default()

    if det is not None and len(det):
        det = det.detach().cpu()
        draw = ImageDraw.Draw(boxed_image)

        print(f"  Detected {len(det)} bounding boxes:")
        for idx, (*box, conf, cls) in enumerate(reversed(det)):
            x1, y1, x2, y2 = [float(i.item()) for i in box]
            class_id = int(cls)
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            color = get_color_for_class(class_id)

            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale

            x1 = max(0.0, min(orig_w, x1))
            y1 = max(0.0, min(orig_h, y1))
            x2 = max(0.0, min(orig_w, x2))
            y2 = max(0.0, min(orig_h, y2))

            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            text_str = f"{class_name} {float(conf):.2f}"

            if hasattr(font, "getbbox"):
                bbox = font.getbbox(text_str)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            else:
                try:
                    text_w, text_h = draw.textsize(text_str, font=font)
                except AttributeError:
                    text_w, text_h = 50, 10

            bg_y1 = max(0, y1 - text_h - 4)
            bg_y2 = bg_y1 + text_h + 4
            draw.rectangle([x1, bg_y1, x1 + text_w + 4, bg_y2], fill=color)
            draw.text((x1 + 2, bg_y1), text_str, fill=(255, 255, 255), font=font)

            print(
                f"  {idx + 1}. Class_ID: {class_id} | Label: {class_name} | "
                f"Confidence: {float(conf) * 100:.1f}% | "
                f"Coords: [x1:{int(x1)}, y1:{int(y1)}, x2:{int(x2)}, y2:{int(y2)}]"
            )

        if args.output:
            boxed_image.save(args.output)
            print(f"[+] Saved image with boxes to: {args.output}")
    else:
        print("  !!! Không tìm thấy đối tượng hợp lệ đáp ứng ngưỡng Confidence.")
        if score_summary["max_conf"] < args.conf:
            print(
                f"  [!] Max confidence raw ({score_summary['max_conf']:.4f}) "
                f"is below --conf {args.conf:.4f}. "
                "Try --conf 0.05 or --conf 0.001 to check score calibration."
            )
        else:
            print(
                "  [!] Some raw scores pass the threshold, but no boxes remain after NMS. "
                "Try --nms-backend torchvision or --nms-backend ultralytics to compare decoding."
            )
    print("==================================================\n")

    if attn_maps and args.heatmap_output:
        scale_name = args.heatmap_scale
        if scale_name in attn_maps:
            attn_weights, (h, w) = attn_maps[scale_name]

            attn = attn_weights[0].max(dim=-1).values
            attn = attn.view(h, w).detach().cpu().numpy()
            attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)

            letterbox_size = int(config["data"]["image_size"])
            attn = cv2.resize(attn, (letterbox_size, letterbox_size), interpolation=cv2.INTER_LINEAR)

            new_w, new_h = img_meta["new_size"]
            attn = attn[pad_h : pad_h + new_h, pad_w : pad_w + new_w]
            attn = cv2.resize(attn, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            img = np.array(original_image)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.imshow(attn, cmap="jet", alpha=args.heatmap_alpha)
            plt.title("Attention Heatmap")
            plt.axis("off")
            plt.savefig(args.heatmap_output, bbox_inches="tight", pad_inches=0)
            plt.close()

            print(f"[+] Saved attention heatmap: {args.heatmap_output}")
        else:
            print(f"[!] Heatmap scale not found: {scale_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="checkpoints/best.pt", help="Path to trained weights")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--config", type=str, default="src/config/default.yaml")
    parser.add_argument("--data-yaml", type=str, default="", help="Đường dẫn tới data.yaml chứa names")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng Confidence")
    parser.add_argument("--iou", type=float, default=0.45, help="Ngưỡng NMS IoU")
    parser.add_argument("--nms-backend", choices=["auto", "ultralytics", "torchvision"], default="auto", help="NMS backend")
    parser.add_argument("--debug-topk", type=int, default=10, help="Number of top raw confidence scores to print")
    parser.add_argument("--output", type=str, default="predictions.jpg", help="Đường dẫn ảnh xuất kết quả")
    parser.add_argument("--heatmap-output", type=str, default="attention_heatmap.jpg", help="Đường dẫn ảnh heatmap")
    parser.add_argument("--heatmap-scale", type=str, default="P3", help="Scale for heatmap: P3/P4/P5")
    parser.add_argument("--heatmap-alpha", type=float, default=0.5, help="Heatmap alpha (0-1)")
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow checkpoint loading even if model/config do not match exactly",
    )

    args = parser.parse_args()
    main(args)
