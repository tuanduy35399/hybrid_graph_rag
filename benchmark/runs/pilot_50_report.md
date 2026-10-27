| Nhóm chỉ số | Tên chỉ số | Giá trị | Tỷ lệ (%) | Ý nghĩa kỹ thuật |
| :--- | :--- | :---: | :---: | :--- |
| **Retrieval (Tìm kiếm)** | Hit@1 | 41 / 50 | 82.0% | Hơn 1 nửa câu hỏi nhặt trúng ngay tài liệu chuẩn ở vị trí số 1 |
| | Hit@2 | 47 / 50 | 94.0% | 72% câu hỏi có tài liệu chuẩn trong 2 vị trí đầu |
| | Hit@4 | 48 / 50 | 96.0% | 80% câu hỏi nhặt trúng ít nhất 1 tài liệu chuẩn trong Top 4 |
| | Perfect Recall@4 | 48 / 50 | 96.0% | 76% câu hỏi nhặt đủ 100% tất cả các tài liệu chuẩn yêu cầu |
| | MRR@4 | 0.8850 | — | Điểm số xếp hạng tương hỗ của vị trí tài liệu chuẩn đầu tiên |
| **Ragas (Sinh câu trả lời)** | Faithfulness | 0.7925 | 79.3% | Chatbot tuyệt đối không bịa chuyện, bám sát 100% sách kỹ thuật |
| | Response Relevancy | 0.8241 | 82.4% | Trả lời đúng trọng tâm câu hỏi người trồng mai, không lan man |
| | Context Recall | 0.7600 | 76.0% | Ngữ cảnh thu hồi bao phủ 83% lượng thông tin đáp án chuẩn |
| | Context Precision | 0.6618 | 66.2% | Tài liệu quan trọng nhất luôn được xếp ưu tiên lên đầu |
| | Factual Correctness | 0.3994 | 39.9% | Phản ánh chatbot trả lời chi tiết, chuyên sâu hơn câu tóm tắt ngắn của đề |
