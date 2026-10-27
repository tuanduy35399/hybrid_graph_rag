# Báo cáo Đánh giá Benchmark: NATIVERAG

## 1. Thông tin chung (Overview)
- **Hệ thống (System):** `nativerag`
- **Tổng số câu hỏi:** `150`
  - Đã đánh giá (Evaluated): `150`
  - Thiếu kết quả (Missing outputs): `0`
  - Lỗi thực thi (Errors): `0`
- **Judge:** `null`
- **Judge calls:** 1225 (decided 0, unknown 1225)
- **Ghi chú (Notes):**
  - 1 output(s) report citations that exactly mirror their retrieved contexts (first: ['Q018']); citations must be parsed from the answer, not copied from retrieval - check the adapter against benchmark/adapters/citation_parser.py

## 2. Chỉ số chính (Primary Metrics)

| Metric | Chiều tốt | Micro | Macro | 95% CI | n | N/A | Unav |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `evidence_recall` | ↑ | 0.835 | 0.836 | [0.784, 0.885] | 135 | 15 | 0 |
| `evidence_recall@5` | ↑ | 0.835 | 0.836 | [0.784, 0.885] | 135 | 15 | 0 |
| `evidence_recall@2000tok` | ↑ | 0.835 | 0.836 | [0.784, 0.885] | 135 | 15 | 0 |
| `hit_rate` | ↑ | 0.697 | 0.692 | [0.637, 0.755] | 135 | 15 | 0 |
| `hit_all` | ↑ | 0.481 | 0.478 | [0.400, 0.563] | 135 | 15 | 0 |
| `citation_article_accuracy` | ↑ | 0.761 | 0.751 | [0.700, 0.816] | 130 | 2 | 18 |
| `citation_recall` | ↑ | 0.950 | 0.950 | [0.909, 0.981] | 135 | 15 | 0 |
| `correct_abstention` | ↑ | 0.000 | 0.000 | [0.000, 0.000] | 15 | 135 | 0 |
| `latency_ms` | ↓ | 7398.712 | 7343.835 | [7031.470, 7801.038] | 150 | 0 | 0 |

## 3. Chỉ số chẩn đoán (Diagnostic Metrics)

| Metric | Chiều tốt | Micro | Macro | 95% CI | n | N/A | Unav |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mrr` | ↑ | 0.838 | 0.831 | [0.786, 0.889] | 135 | 15 | 0 |
| `hit@5` | ↑ | 0.926 | 0.922 | [0.881, 0.970] | 135 | 15 | 0 |
| `context_precision` | ↑ | 0.352 | 0.347 | [0.319, 0.391] | 135 | 15 | 0 |
| `citation_precision` | ↑ | 0.756 | 0.749 | [0.698, 0.811] | 134 | 2 | 14 |
| `citation_document_accuracy` | ↑ | 0.827 | 0.822 | [0.775, 0.876] | 130 | 2 | 18 |
| `error_rate` | ↓ | 0.000 | 0.000 | [0.000, 0.000] | 150 | 0 | 0 |

## 4. Chi tiết theo danh mục (Breakdown by Category - Micro)

| Metric | `definition` | `obligation` | `sanction_numeric` | `effectiveness_metadata` | `inter_document` | `multi_hop` | `unanswerable` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `evidence_recall` | 0.809 | 0.908 | 0.920 | 0.975 | 0.775 | 0.632 | - |
| `evidence_recall@5` | 0.809 | 0.908 | 0.920 | 0.975 | 0.775 | 0.632 | - |
| `evidence_recall@2000tok` | 0.809 | 0.908 | 0.920 | 0.975 | 0.775 | 0.632 | - |
| `hit_rate` | 0.638 | 0.691 | 0.907 | 0.675 | 0.631 | 0.612 | - |
| `hit_all` | 0.500 | 0.440 | 0.800 | 0.500 | 0.350 | 0.280 | - |
| `citation_article_accuracy` | 0.886 | 0.841 | 0.877 | 0.844 | 0.357 | 0.702 | - |
| `citation_recall` | 1.000 | 0.960 | 0.960 | 1.000 | 0.850 | 0.930 | - |
| `correct_abstention` | - | - | - | - | - | - | 0.000 |
| `latency_ms` | 6886.128 | 7941.957 | 7269.983 | 6530.575 | 6784.359 | 8557.946 | 7435.896 |
