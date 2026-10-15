import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Bộ dữ liệu fallback mặc định khi không tìm thấy file bên ngoài
BENCHMARK_DATASET = [
    {
        "user_input": "Cây mai của tôi bị cháy đầu lá rồi lan dần vào trong thành từng vệt màu nâu xám là bị bệnh gì và chữa thế nào?",
        "reference": (
            "Cây mai có triệu chứng cháy từ chóp lá hoặc mép lá lan vào, vết bệnh màu nâu xám có viền đậm "
            "là dấu hiệu của bệnh cháy lá (do nấm Pestalotia funerea gây ra). Cách xử lý: Cắt tỉa và tiêu hủy "
            "các lá bị bệnh nặng để tránh lây lan, phun thuốc trừ nấm gốc đồng như Coc 85, Champion hoặc thuốc "
            "đặc trị như Anvil, Score, Ridomil Gold theo đúng liều lượng hướng dẫn. Đồng thời điều chỉnh không tưới "
            "nước lên lá vào chiều tối và tăng cường phân kali giúp lá dày khỏe."
        ),
        "reference_contexts": [
            (
                "Bệnh cháy lá trên mai vàng do nấm Pestalotia funerea gây ra. Triệu chứng ban đầu xuất hiện ở chóp lá "
                "hoặc mép lá, sau đó lan rộng vào trong phiến lá tạo thành các vệt khô màu xám tro hoặc nâu xám, "
                "có đường viền rõ rệt phân cách giữa phần mô bệnh và mô lành. Trên bề mặt vết bệnh có thể xuất hiện "
                "các hạt nhỏ màu đen là ổ bào tử nấm."
            ),
            (
                "Biện pháp phòng trừ bệnh cháy lá mai vàng: Tỉa bớt cành lá rậm rạp, thu gom lá bệnh đem tiêu hủy. "
                "Sử dụng các loại thuốc gốc đồng như Booc-đô, Coc 85, hoặc các hoạt chất như Mancozeb, Hexaconazole (Anvil), "
                "Difenoconazole (Score) phun ướt đều tán lá. Bón bổ sung lân và kali, hạn chế bón thừa phân đạm khi bệnh đang phát triển."
            )
        ]
    },
    {
        "user_input": "mai bi dom nau tron tren la non la bi lam sao",
        "reference": (
            "Hiện tượng xuất hiện các đốm tròn màu nâu hoặc nâu sẫm trên lá non của mai vàng là triệu chứng của "
            "bệnh thán thư (do nấm Colletotrichum gloeosporioides gây ra). Cách điều trị: Thu gom tiêu hủy lá bị hại, "
            "tránh tưới đẫm nước vào tán lá non vào ban đêm. Phun luân phiên các loại thuốc đặc trị thán thư như Antracol 70WP, "
            "Amistar Top, Score 250EC hoặc Daconil 75WP theo liều lượng hướng dẫn trên bao bì."
        ),
        "reference_contexts": [
            (
                "Bệnh thán thư do nấm Colletotrichum gloeosporioides gây hại chủ yếu trên lá non, chồi non của cây mai. "
                "Vết bệnh ban đầu là những đốm nhỏ hình tròn màu nâu đỏ hoặc nâu sẫm, sau đó vết bệnh lớn dần có thể làm "
                "thủng rách phiến lá, méo mó lá non và rụng sớm."
            ),
            (
                "Phòng trị bệnh thán thư trên mai vàng: Không để cây quá rậm rạp, phun phòng ngừa khi cây bung đọt non bằng "
                "Antracol 70WP. Khi thấy xuất hiện vết bệnh, sử dụng các thuốc nội hấp lưu dẫn như Amistar Top, Score, Tilt Super "
                "phun định kỳ 7-10 ngày một lần cho đến khi đợt đọt non thuần thục."
            )
        ]
    },
    {
        "user_input": "Trên thân cành cây mai xuất hiện lớp phấn màu hồng nhạt rồi cành khô héo dần, đó là bệnh gì?",
        "reference": (
            "Đó là bệnh nấm hồng (do nấm Corticium salmonicolor gây ra). Nấm thường tấn công ở các chạc ba hoặc vỏ thân, "
            "cành cây nơi ẩm ướt, tạo lớp màng mỏng màu trắng rồi chuyển sang hồng phấn, làm vỏ cây nứt và tắc mạch dẫn "
            "khiến cành khô chết. Cách trị: Dùng bàn chải mềm chà sạch lớp nấm, quét thuốc gốc đồng đậm đặc như Champion, "
            "Coc 85 hoặc phun thuốc hoạt chất Validamycin (Validacin), Hexaconazole (Anvil) lên toàn bộ thân cành."
        ),
        "reference_contexts": [
            (
                "Bệnh nấm hồng do nấm Corticium salmonicolor phát triển mạnh trong mùa mưa, độ ẩm không khí cao. "
                "Ban đầu vỏ thân cành xuất hiện tơ nấm trắng, sau đó dày lên thành lớp phấn màu hồng nhạt hoặc đỏ cam. "
                "Nấm ăn sâu vào lớp mô vỏ làm thâm đen mạch dẫn, khiến lá phía trên cành bị héo vàng và cành chết khô."
            ),
            (
                "Xử lý bệnh nấm hồng: Cắt bỏ các cành khô chết nặng đem đốt. Cạo sạch vết nấm trên thân chính rồi quét "
                "dung dịch thuốc Bordeaux, Coc 85 hoặc Validacin đặc lên vết bệnh. Có thể phun toàn cây bằng thuốc Anvil 5SC "
                "hoặc Tilt Super."
            )
        ]
    }
]


def _normalize_sample(raw_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chuẩn hóa linh hoạt các trường khác nhau trong JSON/JSONL về dạng chuẩn:
    - user_input: câu hỏi
    - reference: đáp án tham chiếu
    - reference_contexts: danh sách context mẫu
    """
    # 1. Lấy câu hỏi
    user_input = (
        raw_item.get("user_input")
        or raw_item.get("question")
        or raw_item.get("query")
        or raw_item.get("prompt")
        or raw_item.get("input")
        or ""
    )

    # 2. Lấy câu trả lời chuẩn (Ground Truth)
    reference = (
        raw_item.get("reference")
        or raw_item.get("reference_answer")
        or raw_item.get("ground_truth")
        or raw_item.get("target")
        or raw_item.get("answer")
        or ""
    )

    # 3. Lấy reference contexts
    raw_contexts = (
        raw_item.get("reference_contexts")
        or raw_item.get("contexts")
        or raw_item.get("context")
        or []
    )

    # Nếu có gold_evidence dạng list of dicts (như trong pilot.jsonl)
    if not raw_contexts and "gold_evidence" in raw_item:
        gold = raw_item["gold_evidence"]
        if isinstance(gold, list):
            raw_contexts = [
                g.get("content", "") if isinstance(g, dict) else str(g)
                for g in gold
            ]

    # Đảm bảo reference_contexts là list of strings
    if isinstance(raw_contexts, str):
        contexts = [raw_contexts]
    elif isinstance(raw_contexts, list):
        contexts = [
            c.get("content", str(c)) if isinstance(c, dict) else str(c)
            for c in raw_contexts
        ]
    else:
        contexts = []

    return {
        "user_input": user_input.strip(),
        "reference": reference.strip(),
        "reference_contexts": contexts,
        "case_id": raw_item.get("case_id", ""),
        "intent_family": raw_item.get("intent_family", ""),
        "tags": raw_item.get("tags", []),
        "raw": raw_item
    }


def load_dataset_flexible(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Nạp bộ dữ liệu đánh giá linh hoạt từ file .jsonl hoặc .json.
    - Nếu truyền file_path: Ưu tiên đọc file đó (tìm ở đường dẫn tuyệt đối hoặc tương đối).
    - Nếu không truyền: Tự động kiểm tra file 'pilot.jsonl' ở thư mục gốc.
    - Nếu không có file nào: Sử dụng BENCHMARK_DATASET có sẵn.
    """
    target_path = None

    if file_path:
        p = Path(file_path)
        if p.exists():
            target_path = p
        elif (PROJECT_ROOT / file_path).exists():
            target_path = PROJECT_ROOT / file_path
        elif (PROJECT_ROOT / "evaluation" / file_path).exists():
            target_path = PROJECT_ROOT / "evaluation" / file_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dataset tại: {file_path}")
    else:
        # Tự động ưu tiên pilot.jsonl nếu có trong thư mục gốc
        default_pilot = PROJECT_ROOT / "pilot.jsonl"
        if default_pilot.exists():
            target_path = default_pilot

    if target_path is None:
        print("[DATASET] Không chỉ định file, sử dụng BENCHMARK_DATASET mặc định.")
        return [_normalize_sample(item) for item in BENCHMARK_DATASET]

    print(f"[DATASET] Đang nạp dữ liệu đánh giá từ: {target_path.name}")
    samples = []

    if target_path.suffix.lower() == ".jsonl":
        with open(target_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    samples.append(_normalize_sample(item))
                except json.JSONDecodeError as err:
                    print(f"[CẢNH BÁO] Lỗi cú pháp JSON dòng {line_no} trong {target_path.name}: {err}")

    elif target_path.suffix.lower() == ".json":
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    samples.append(_normalize_sample(item))
            elif isinstance(data, dict):
                # Một số dataset có dạng {"data": [...]} hoặc {"samples": [...]}
                list_data = data.get("data") or data.get("samples") or [data]
                for item in list_data:
                    samples.append(_normalize_sample(item))
    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {target_path.suffix}. Vui lòng dùng .jsonl hoặc .json.")

    print(f"[DATASET] Nạp thành công {len(samples)} mẫu từ {target_path.name}!")
    return samples
