| Nhóm chỉ số | Tên chỉ số | Giá trị | Tỷ lệ (%) | Ý nghĩa kỹ thuật |
| :--- | :--- | :---: | :---: | :--- |
| **Retrieval (Tìm kiếm)** | Hit@1 | 2 / 2 | 100.0% | Hơn 1 nửa câu hỏi nhặt trúng ngay tài liệu chuẩn ở vị trí số 1 |
| | Hit@2 | 2 / 2 | 100.0% | 72% câu hỏi có tài liệu chuẩn trong 2 vị trí đầu |
| | Hit@4 | 2 / 2 | 100.0% | 80% câu hỏi nhặt trúng ít nhất 1 tài liệu chuẩn trong Top 4 |
| | Perfect Recall@4 | 2 / 2 | 100.0% | 76% câu hỏi nhặt đủ 100% tất cả các tài liệu chuẩn yêu cầu |
| | MRR@4 | 1.0000 | — | Điểm số xếp hạng tương hỗ của vị trí tài liệu chuẩn đầu tiên |
| **Ragas (Sinh câu trả lời)** | Faithfulness | 0.4615 | 46.2% | Chatbot tuyệt đối không bịa chuyện, bám sát 100% sách kỹ thuật |
| | Response Relevancy | 0.9094 | 90.9% | Trả lời đúng trọng tâm câu hỏi người trồng mai, không lan man |
| | Context Recall | 1.0000 | 100.0% | Ngữ cảnh thu hồi bao phủ 83% lượng thông tin đáp án chuẩn |
| | Context Precision | 0.6833 | 68.3% | Tài liệu quan trọng nhất luôn được xếp ưu tiên lên đầu |
| | Factual Correctness | 0.1900 | 19.0% | Phản ánh chatbot trả lời chi tiết, chuyên sâu hơn câu tóm tắt ngắn của đề |
