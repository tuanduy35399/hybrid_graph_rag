# Báo cáo Đánh giá Benchmark: HYBRIDRAG

## 1. Thông tin chung (Overview)
- **Hệ thống (System):** `hybridrag`
- **Tổng số câu hỏi:** `50`
  - Đã đánh giá (Evaluated): `2`
  - Thiếu kết quả (Missing outputs): `48`
  - Lỗi thực thi (Errors): `0`
- **Judge:** `null`
- **Ghi chú (Notes):**
  - 48 question(s) had no system output and were excluded from every mean (not scored as 0)

## 2. Chỉ số chính (Primary Metrics)

| Metric | Chiều tốt | Micro | Macro | 95% CI | n | N/A | Unav |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `evidence_recall` | ↑ | 1.000 | 1.000 | [1.000, 1.000] | 2 | 0 | 0 |
| `evidence_recall@5` | ↑ | 1.000 | 1.000 | [1.000, 1.000] | 2 | 0 | 0 |
| `evidence_recall@2000tok` | ↑ | 1.000 | 1.000 | [1.000, 1.000] | 2 | 0 | 0 |
| `hit_rate` | ↑ | - | - | - | 0 | 2 | 0 |
| `hit_all` | ↑ | - | - | - | 0 | 2 | 0 |
| `citation_article_accuracy` | ↑ | - | - | - | 0 | 2 | 0 |
| `citation_recall` | ↑ | 0.000 | 0.000 | [0.000, 0.000] | 2 | 0 | 0 |
| `correct_abstention` | ↑ | - | - | - | 0 | 2 | 0 |
| `latency_ms` | ↓ | 21207.847 | 21207.847 | [20239.666, 22176.027] | 2 | 0 | 0 |

## 3. Chỉ số chẩn đoán (Diagnostic Metrics)

| Metric | Chiều tốt | Micro | Macro | 95% CI | n | N/A | Unav |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mrr` | ↑ | 1.000 | 1.000 | [1.000, 1.000] | 2 | 0 | 0 |
| `hit@5` | ↑ | 1.000 | 1.000 | [1.000, 1.000] | 2 | 0 | 0 |
| `context_precision` | ↑ | 0.900 | 0.900 | [0.800, 1.000] | 2 | 0 | 0 |
| `citation_precision` | ↑ | - | - | - | 0 | 2 | 0 |
| `citation_document_accuracy` | ↑ | - | - | - | 0 | 2 | 0 |
| `error_rate` | ↓ | 0.000 | 0.000 | [0.000, 0.000] | 2 | 0 | 0 |

## 4. Chi tiết theo danh mục (Breakdown by Category - Micro)

| Metric | `care_guidance` |
| :--- | :---: |
| `evidence_recall` | 1.000 |
| `evidence_recall@5` | 1.000 |
| `evidence_recall@2000tok` | 1.000 |
| `hit_rate` | - |
| `hit_all` | - |
| `citation_article_accuracy` | - |
| `citation_recall` | 0.000 |
| `correct_abstention` | - |
| `latency_ms` | 21207.847 |
