# Báo cáo Đánh giá Benchmark: HYBRIDRAG

## 1. Thông tin chung (Overview)
- **Hệ thống (System):** `hybridrag`
- **Tổng số câu hỏi:** `150`
  - Đã đánh giá (Evaluated): `150`
  - Thiếu kết quả (Missing outputs): `0`
  - Lỗi thực thi (Errors): `0`
- **Judge:** `null`
- **Judge calls:** 733 (decided 0, unknown 733)

## 2. Chỉ số chính (Primary Metrics)

| Metric | Chiều tốt | Micro | Macro | 95% CI | n | N/A | Unav |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `evidence_recall` | ↑ | 0.848 | 0.846 | [0.793, 0.902] | 135 | 15 | 0 |
| `evidence_recall@5` | ↑ | 0.848 | 0.846 | [0.793, 0.902] | 135 | 15 | 0 |
| `evidence_recall@2000tok` | ↑ | 0.848 | 0.846 | [0.793, 0.902] | 135 | 15 | 0 |
| `hit_rate` | ↑ | 0.678 | 0.675 | [0.625, 0.736] | 135 | 15 | 0 |
| `hit_all` | ↑ | 0.415 | 0.413 | [0.341, 0.496] | 135 | 15 | 0 |
| `citation_article_accuracy` | ↑ | 0.626 | 0.613 | [0.575, 0.681] | 131 | 2 | 17 |
| `citation_recall` | ↑ | 0.970 | 0.968 | [0.941, 0.993] | 135 | 15 | 0 |
| `correct_abstention` | ↑ | 0.133 | 0.133 | [0.000, 0.333] | 15 | 135 | 0 |
| `latency_ms` | ↓ | 37708.754 | 37481.577 | [36744.350, 38839.038] | 150 | 0 | 0 |

## 3. Chỉ số chẩn đoán (Diagnostic Metrics)

| Metric | Chiều tốt | Micro | Macro | 95% CI | n | N/A | Unav |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mrr` | ↑ | 0.864 | 0.860 | [0.807, 0.915] | 135 | 15 | 0 |
| `hit@5` | ↑ | 0.911 | 0.905 | [0.859, 0.956] | 135 | 15 | 0 |
| `context_precision` | ↑ | 0.415 | 0.407 | [0.376, 0.456] | 135 | 15 | 0 |
| `citation_precision` | ↑ | 0.677 | 0.666 | [0.629, 0.728] | 133 | 2 | 15 |
| `citation_document_accuracy` | ↑ | 0.810 | 0.805 | [0.757, 0.856] | 130 | 2 | 18 |
| `error_rate` | ↓ | 0.000 | 0.000 | [0.000, 0.000] | 150 | 0 | 0 |

## 4. Chi tiết theo danh mục (Breakdown by Category - Micro)

| Metric | `definition` | `obligation` | `sanction_numeric` | `effectiveness_metadata` | `inter_document` | `multi_hop` | `unanswerable` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `evidence_recall` | 0.863 | 0.966 | 0.973 | 0.950 | 0.675 | 0.651 | - |
| `evidence_recall@5` | 0.863 | 0.966 | 0.973 | 0.950 | 0.675 | 0.651 | - |
| `evidence_recall@2000tok` | 0.863 | 0.966 | 0.973 | 0.950 | 0.675 | 0.651 | - |
| `hit_rate` | 0.649 | 0.633 | 0.907 | 0.658 | 0.640 | 0.563 | - |
| `hit_all` | 0.400 | 0.320 | 0.760 | 0.450 | 0.350 | 0.200 | - |
| `citation_article_accuracy` | 0.614 | 0.724 | 0.799 | 0.666 | 0.301 | 0.571 | - |
| `citation_recall` | 1.000 | 0.960 | 1.000 | 1.000 | 0.850 | 1.000 | - |
| `correct_abstention` | - | - | - | - | - | - | 0.133 |
| `latency_ms` | 37594.695 | 39867.012 | 37743.619 | 36361.452 | 40097.193 | 37437.453 | 33269.614 |
