# Báo Cáo Đánh Giá Hệ Thống GraphRAG Cây Mai Vàng

- **Thời điểm đánh giá:** 2026-09-15 18:35:47
- **Nguồn dữ liệu đánh giá:** `pilot.jsonl`
- **Số lượng mẫu kiểm thử:** 20 mẫu
- **Engine đánh giá:** `heuristic_fallback`

## 1. Điểm số Chất lượng RAG (RAG Quality Metrics)

| Chỉ số đánh giá | Điểm số (0.0 - 1.0) | Ý nghĩa |
| :--- | :--- | :--- |
| **Faithfulness** | `0.9572` | Câu trả lời bám sát context, không bịa đặt nguyên nhân/thuốc |
| **Answer Relevancy** | `1.0000` | Câu trả lời phản hồi đúng trọng tâm câu hỏi |
| **Context Precision** | `0.9758` | Ngữ cảnh được truy xuất chứa đúng thông tin trọng điểm |
| **Context Recall** | `0.9750` | Ngữ cảnh bao quát đủ thông tin theo đáp án chuẩn tham chiếu |

## 2. Thống kê Tốc độ Truy vấn Từng Giai đoạn (Latency Breakdown)

| Giai đoạn trong Pipeline | Trung bình (s) | Trung bình (ms) | P95 (s) | Tối đa (s) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Query Rewrite** | 6.0674s | 6067.4ms | 13.2764s | 13.2764s |
| **2. HyDE Generation** | 11.0525s | 11052.5ms | 13.7465s | 13.7465s |
| **3. Vector Search (Chroma)** | 1.4156s | 1415.6ms | 4.4735s | 4.4735s |
| **4. Graph Search (Neo4j)** | 6.3740s | 6374.0ms | 8.1207s | 8.1207s |
| **5. Hybrid Merge** | 0.0000s | 0.0ms | 0.0000s | 0.0000s |
| **6. Cross-Encoder Rerank** | 12.2341s | 12234.1ms | 22.9474s | 22.9474s |
| **7. LLM Answer Gen** | 15.1001s | 15100.1ms | 21.8587s | 21.8587s |
| **TỔNG TOÀN BỘ PIPELINE** | 52.2438s | 52243.8ms | 61.6269s | 61.6269s |

## 3. Chi tiết Từng Mẫu Kiểm Thử

### Mẫu 1 [care_nam_hong_01]: Cách trị bệnh nấm hồng thế nào?
- **Phân loại intent:** `care_guidance` | **Tags:** `disease, fungicide, pink-fungus`
- **Tổng thời gian xử lý:** `60.9747s`
- **Câu trả lời sinh ra:**
> Chào bạn, dưới đây là hướng dẫn chi tiết về dấu hiệu, nguyên nhân và các bước xử lý bệnh nấm hồng trên cây mai vàng dựa trên tài liệu:

### 1. Dấu hiệu và tác hại của bệnh
* **Vị trí xuất hiện:** Nấm thường phát triển trên thân, cành, chi và lá mai.
* **Tác hại:** Bệnh khiến vỏ cây bị nứt nẻ, lột vỏ, làm hư hại toàn bộ lá trên nhánh bị bệnh, gây hiện tượng bỏ chi (chết cành), cành bị khô chết và khiến cây suy kiệt rất nhanh.

### 2. Nguyên nhân và điều kiện thuận lợi
* **Thời tiết và môi trường:** Mùa mưa, thời tiết ẩm ướt, độ ẩm cao hoặc vườn cây thiếu thông thoáng, tàng lá quá um tùm và rậm rạp.
* **Giống cây:** Một số giống mai như mai vàng zin, dảo xoắn, dảo cam, dảo cúc cam... thường nhạy cảm và dễ nhiễm nấm hồng hơn.

---

### 3. Quy trình các bước trị bệnh nấm hồng

* **Bước 1: Pha thuốc quét trực tiếp**
  * Sử dụng một trong các loại thuốc sau:
    * **COC85:** Pha tỷ lệ khoảng 10g thuốc với 50ml – 100ml nước (thích hợp dùng vào mùa mưa vì bám dính tốt).
    * **CHAMPION 77WP:** Dùng hiệu quả cao vào mùa nắng.
    * **Ridomil Gold:** Pha tỷ lệ 10g thuốc với 50ml nước.

* **Bước 2: Quét thuốc lên vết bệnh**
  * Dùng cọ quét, bàn chải hoặc khăn vải thấm đều dung dịch thuốc đã pha, sau đó quét trực tiếp lên các vị trí thân, cành đang bị nấm hồng bám.

* **Bước 3: Kiểm tra và xử lý tiếp theo (sau 5 – 10 ngày)**
  * Sau khi quét từ 5 đến 10 ngày, quan sát kỹ vết nấm:
    * *Nếu nấm chưa chết hẳn:* Tiếp tục pha thuốc và quét lần thứ hai.
    * *Nếu nấm đã khô cháy và chết:* Tiến hành pha thuốc nấm (như **Anvil** hoặc **Ridomil Gold**) phun xịt ướt đẫm toàn bộ thân, cành, nhánh và lá để tiêu diệt mầm bệnh còn sót lại và phòng ngừa tái phát. Lịch phun ngừa định kỳ tiếp theo có thể cách nhau 20 ngày/lần.

* **Bước 4: Lưu ý chăm sóc và dọn dẹp cành lá bệnh**
  * **Đối với lá:** Không nên lặt bỏ lá bệnh tùy tiện (chỉ lặt lá vào đợt tỉa giữa năm hoặc tuốt lá đón Tết), vì lá vẫn hỗ trợ cây quang hợp nuôi thân.
  * **Đối với cành:** Nếu cành bị bệnh không ảnh hưởng đến dáng tàng của cây, có thể cắt bỏ rồi đem đi tiêu hủy (đốt hoặc gom vào bao nilon vứt xa vườn).
  * Thường xuyên theo dõi và xịt thuốc phòng ngừa định kỳ ướt đẫm toàn cây, nhất là với những cây đã từng bị nấm hồng.

---

**Một số câu hỏi gợi ý để hỗ trợ bạn rõ hơn:**
1. Cây mai của bạn hiện đang bị nấm hồng ở thân gốc hay trên các cành nhánh nhỏ?
2. Vườn mai của bạn đang ở mùa mưa hay mùa nắng để chọn loại thuốc quét cho phù hợp nhất?

- **Đáp án tham chiếu (Ground Truth):**
> Pha COC85 theo liều nêu trong tài liệu, quét hỗn hợp lên thân hoặc cành bị nấm hồng; sau 5–10 ngày kiểm tra lại. Khi nấm đã cháy/chết, xịt thuốc nấm lên thân, cành, lá để phòng tái phát.

### Mẫu 2 [care_ridomil_gold_01]: Cách trị dùng thuốc Ridomil Gold?
- **Phân loại intent:** `care_guidance` | **Tags:** `fungicide, ridomil-gold, root-care`
- **Tổng thời gian xử lý:** `60.1435s`
- **Câu trả lời sinh ra:**
> Thuốc **Ridomil Gold** (hoạt chất Metalaxyl + Mancozeb) là dòng thuốc trừ nấm nội hấp, lưu dẫn mạnh, dùng để đặc trị và phòng ngừa các bệnh như thối rễ, xì mủ, nấm hồng, địa y, thối nhũn trên cây mai vàng.

Dưới đây là hướng dẫn chi tiết các cách sử dụng thuốc Ridomil Gold cho từng trường hợp:

---

### 1. Quét trực tiếp lên thân, cành
* **Khi trị bệnh (nấm hồng, địa y, vết xì mủ):** 
  * Pha **10g thuốc với 50ml nước** (hoặc 50–100ml nước).
  * Dùng cọ hoặc bàn chải quét đẫm trực tiếp lên khu vực bị bệnh. 
  * Sau khoảng 5–10 ngày, tiến hành xịt phủ Ridomil Gold (hoặc Anvil) ướt đẫm toàn bộ thân - cành - lá để diệt sạch mầm bệnh và hạn chế tái phát.
* **Khi quét phòng bệnh định kỳ:** 
  * Pha **10g thuốc với 100ml nước**, quét lên thân vào 2 thời điểm: sau dịp Tết và thời điểm trước/sau khi dứt mùa mưa.

---

### 2. Phun xịt lên thân và bộ lá
* **Khi trị bệnh:** Phun định kỳ từ **10–15 ngày/lần** (thường luân chuyển sau khi đã xịt chặn bằng COC85). Lưu ý thuốc chỉ ngăn chặn bệnh lây lan chứ không làm mất vết sẹo cũ.
* **Khi phòng bệnh:** 
  * Liều lượng: Pha **1–2g/lít nước**.
  * Phun lần đầu khi lá bước vào cơi 2 đã chuyển xanh. 
  * Vào mùa mưa, xịt 1 tháng/lần; đợt cao điểm mưa nhiều thì xịt xen kẽ Ridomil Gold với Anvil khoảng 20 ngày/lần.

---

### 3. Xử lý nấm rễ, ngâm và phục hồi rễ (thối rễ, xì mủ)
* **Xử lý rễ bị nấm:** Pha **2g/lít nước** (kết hợp với Atonik liều 1ml/2 lít nước; nếu có tuyến trùng thì pha thêm Tervigo).
* **Cách làm:** Rũ bỏ toàn bộ đất cũ, cắt tỉa hết các đoạn rễ bị thối mục. Sau đó ngâm bộ rễ (hoặc quấn vải đẫm thuốc quanh thân đối với cây bị nặng) trong dung dịch thuốc từ **8–12 tiếng** trước khi vô chậu mới.
* **Tưới phục hồi:** Khi trồng lại, tưới đẫm dung dịch trên vào chất trồng, trùm kín bọc nilon để nơi râm mát. Định kỳ khoảng 15 ngày/lần tưới lại hỗn hợp này (làm khoảng 3 lần).
* **Tưới phòng định kỳ vào chất trồng:** Thực hiện sau Tết (khoảng tháng 1 âm lịch) và trước mùa mưa, tưới định kỳ 2–3 tuần/lần (luân phiên cùng Anvil hoặc Coc85).

---

### ⚠️ Lưu ý quan trọng khi sử dụng:
* **Thuốc có tính nóng mạnh:** Không được lạm dụng hoặc tưới gốc quá thường xuyên (chỉ nên tưới tối đa **1 lần/tháng**). Lạm dụng sẽ gây hiện tượng "nóng rễ", làm cháy rễ non và vàng đồng loạt hai bên mặt lá khắp toàn cây.
* **Xử lý khi bị nóng rễ do quá liều:** Phải dừng thuốc ngay, giảm liều pha (xuống còn 1g/lít), giãn cách thời gian tưới và có thể dùng **Vitamin B12** (1000mcg pha 0.5 lít nước) để giải độc cho cây.
* **Luân phiên thuốc:** Cần thay đổi qua lại với các loại thuốc nấm khác (như Anvil, Coc85, Antracol) để tránh hiện tượng nấm bị kháng thuốc.

---

**Một số câu hỏi mở rộng để hỗ trợ bác tốt hơn:**
1. Cây mai của bác hiện đang có dấu hiệu bệnh cụ thể ở rễ, thân hay trên lá?
2. Bác đang dự định dùng Ridomil Gold để xử lý trị bệnh cụ thể nào hay chỉ muốn phun/tưới để phòng ngừa định kỳ?

- **Đáp án tham chiếu (Ground Truth):**
> Tài liệu nêu Ridomil Gold là thuốc nấm có thể pha loãng để quét lên thân hoặc xịt lá. Một hướng dẫn cụ thể là tưới gốc định kỳ với liều 2 gram/1 lít nước, tưới đẫm đầu năm khi thay hoặc thêm chất trồng.

### Mẫu 3 [care_nam_re_01]: Phòng bệnh nấm rễ cho mai như thế nào?
- **Phân loại intent:** `care_guidance` | **Tags:** `fungus, prevention, root-care`
- **Tổng thời gian xử lý:** `55.6046s`
- **Câu trả lời sinh ra:**
> Để phòng bệnh nấm rễ cho cây mai vàng một cách hiệu quả, bà con nên kết hợp giữa việc dùng thuốc phòng ngừa định kỳ và quản lý tốt môi trường chất trồng như sau:

---

### 1. Tưới thuốc phòng nấm định kỳ
* **Loại thuốc sử dụng:** Có thể sử dụng các loại thuốc chuyên dùng tưới gốc như **Ridomil Gold** hoặc men vi sinh **Trichoderma** (hoặc các loại thuốc nấm tưới gốc tương đương).
* **Thời điểm cần tưới phòng:**
  * **Khi thay hoặc thêm chất trồng mới:** Ngay sau khi vào chậu mới hoặc thêm chất trồng, cần tưới một đợt thuốc nấm để tiêu diệt các mầm bệnh còn tồn đọng trong giá thể.
  * **Vào mùa mưa:** Đây là thời điểm độ ẩm cao, nấm dễ phát sinh nhất. Bà con nên tưới phòng định kỳ vào **3 đợt: Đầu mùa mưa, giữa mùa mưa và cuối mùa mưa**.

---

### 2. Quản lý giá thể và khả năng thoát nước (Ngăn ngừa môi trường sinh nấm)
Nấm rễ phát sinh chủ yếu do đất ẩm thấp kéo dài, bí khí, ứ nước. Do đó, cần lưu ý:
* **Đảm bảo chậu thoát nước tốt:** Đáy chậu phải có lỗ thoát nước thông thoáng, nên lót vật liệu thoáng khí ở đáy chậu trước khi vô cây để tránh đọng nước, lèn sình gây ngộp rễ.
* **Kiểm soát lượng nước tưới:** Không tưới quá nhiều lần liên tục; chỉ tưới khi kiểm tra thấy độ ẩm phù hợp, tránh để chất trồng luôn sũng nước.
* **Tránh để chất trồng bị nén chặt, thiếu oxy:** Sau một thời gian, chất trồng phân hủy dễ bị bết dính. Hàng năm nên xăm và bổ sung chất trồng mới (nếu không thay toàn bộ chậu) để rễ luôn thoáng khí.
* **Chống bọng/sụp chất trồng:** Khoảng 15–20 ngày nên tưới thật đẫm nước cho chảy đáy 30–60 giây để chất trồng dẻ đều, tránh rễ bị hở dẫn đến tổn thương và tạo cơ hội cho nấm tấn công.

---

**Một số câu hỏi để tiện hỗ trợ thêm cho bà con:**
1. Cây mai của bà con hiện đang được trồng bằng loại chất trồng nào (xơ dừa, trấu, đất...) và chậu thoát nước có tốt không?
2. Cây của bà con đang xanh tốt muốn phòng bệnh định kỳ, hay cây đã có dấu hiệu vàng lá, đứng đọt nghi bị nấm rễ rồi?

- **Đáp án tham chiếu (Ground Truth):**
> Dùng chất trồng thoát nước tốt, không giữ ẩm quá lâu; tránh nén chặt để chất trồng tơi xốp và nhiều oxy. Có thể trộn hoặc dùng định kỳ nấm đối kháng Trichoderma để ức chế nấm hại.

### Mẫu 4 [care_tuoi_nuoc_01]: Tưới nước cho mai phôi thế nào để không úng rễ?
- **Phân loại intent:** `care_guidance` | **Tags:** `mai-phoi, root-care, watering`
- **Tổng thời gian xử lý:** `55.8352s`
- **Câu trả lời sinh ra:**
> Để tưới nước cho cây mai phôi đúng cách và không làm cây bị úng rễ, bà con nông dân cần lưu ý không áp dụng máy móc theo số lít nước hay số ngày cố định, mà phải theo dõi và làm theo các bước sau:

---

### 1. Nguyên tắc tưới nước cho cây mai phôi
* **Lần đầu khi mới vô chậu:** Tưới đẫm nước một lần để tạo đường dẫn nước (mạch nước) cho chất trồng. Lần tưới này có thể giữ ẩm từ vài ngày đến nửa tháng tùy kích thước chậu và loại giá thể.
* **Giữ ẩm hàng ngày:** Công việc mỗi ngày là dùng vòi xịt phun sương để làm ẩm thân cây (và mặt chậu nếu để ngoài nắng) vào lúc **sáng sớm và chiều tối**, tuyệt đối không tưới đẫm gốc liên tục.
* **Tưới phả mặt chậu:** Sau khoảng **3 đến 15 ngày**, bà con dùng ngón tay kiểm tra độ ẩm (thọt sâu xuống khoảng 1 ngón tay). Khi thấy chất trồng đã hết ẩm ướt và bắt đầu sắp khô thì mới dùng vòi béc xịt lướt nhẹ, nhanh qua mặt chậu.
* **Tưới đẫm định kỳ:** Cách khoảng **10 đến 30 ngày** (tùy chậu lớn hay nhỏ), cần tưới đẫm nước một lần cho nước chảy tràn qua lỗ đáy chậu. Việc này giúp toàn bộ chất trồng được ẩm đều và đẩy hết lượng "nước cũ" tù đọng ra ngoài, tránh nấm bệnh tấn công rễ.

---

### 2. Cách kiểm tra để phát hiện cây bị dư nước / úng rễ
* **Kiểm tra mặt chậu:** Thọt ngón tay xuống lớp chất trồng sâu khoảng 1 ngón tay. Nếu sau 2 - 3 ngày tưới mà chất trồng vẫn còn sũng nước thì cây đang bị dư nước.
* **Kiểm tra đáy chậu:** Luồn tay vào lỗ thoát nước dưới đáy chậu, nếu thấy sình, bùn ứ đọng và ẩm ướt kéo dài thì chậu đang bị úng nước ở đáy, rễ cây sẽ thiếu oxy để "thở" và dễ bị thúi rễ.
* **Yếu tố ảnh hưởng:** Trời nắng chất trồng mau khô hơn trời mưa; chậu nhỏ mau khô hơn chậu lớn; chất trồng giữ nước (như xơ dừa, cám dừa) sẽ lâu khô hơn. Do đó cần linh hoạt kiểm tra thường xuyên.

---

**Một số câu hỏi để hỗ trợ bà con tốt hơn:**
1. Cây mai phôi của bà con hiện đang được trồng bằng loại chất trồng nào (xơ dừa, trấu hay đất...) và kích thước chậu lớn hay nhỏ?
2. Cây mới vô chậu hay đã bắt đầu nảy chồi non rồi?
3. Nguồn nước bà con đang dùng để tưới là nước máy, nước giếng khoan hay nước sông/kênh rạch?

- **Đáp án tham chiếu (Ground Truth):**
> Không có một công thức cố định về lượng và số ngày tưới. Sau lần tưới đẫm đầu, giữ ẩm thân và mặt chậu; sau 3–15 ngày, kiểm tra độ ẩm chất trồng rồi mới tưới phả khi sắp khô.

### Mẫu 5 [care_bon_phan_01]: Khi nào nên bón phân cho cây mai?
- **Phân loại intent:** `care_guidance` | **Tags:** `fertilizer, grafted-tree, timing`
- **Tổng thời gian xử lý:** `53.0217s`
- **Câu trả lời sinh ra:**
> Thời điểm bón phân cho cây mai phụ thuộc vào từng loại cây (mai mới ghép hay mai thành phẩm) và từng giai đoạn phát triển trong năm:

### 1. Đối với mai mới ghép (mai phôi)
* **Không bón phân quá sớm:** Giai đoạn mới ghép cây còn rất yếu, tuyệt đối không nên vội bón phân.
* **Thời điểm bắt đầu:** Sau khi tháo bọc ít nhất **03 tháng** mới nên bắt đầu cho cây ăn phân hữu cơ. (Trường hợp cây phôi đã thuần chậu khỏe mạnh thì có thể cho ăn phân sớm hơn một chút).

---

### 2. Đối với cây mai thành phẩm (theo lịch trình trong năm)
* **Tháng 1 – Tháng 2 (Giai đoạn hồi phục):** Chưa bón phân NPK/hóa học, chỉ tập trung tưới dưỡng rễ (định kỳ 10–15 ngày/lần) và dùng chất kích thích sinh trưởng 1–2 lần tùy độ mạnh của cây.
* **Tháng 3:** Khi cây bắt đầu **ra cơi lá thứ 2**, lúc này mới bắt đầu bón phân hữu cơ (như phân Dynamic hoặc phân bánh dầu) kèm dưỡng rễ.
* **Tháng 4 – Tháng 6 (Giai đoạn phát triển cành lá):** Bắt đầu nạp phân vô cơ NPK 30-10-10, sau đó khoảng 20 ngày có thể dùng kèm DAP; kết hợp bón phân hữu cơ và tưới dưỡng rễ luân phiên (1–2 lần/tháng).
* **Tháng 7 – Tháng 9 (Giai đoạn làm nút/tạo nụ):** Bón NPK 21-21-21 để tạo nút; nếu cây chưa đạt kim có thể bổ sung DAP hoặc NPK 10-60-10; duy trì phân hữu cơ và dưỡng rễ đều đặn.
* **Tháng 10 – Tháng 11 (Giai đoạn nuôi và làm già nút):** Sử dụng NPK 21-21-21, NPK 6-32-32, phân Kali đỏ, Canxi-Bo và phân hữu cơ (lưu ý không dùng phân dơi).
* **Tháng 12:** Thường **ngưng bón phân**, chỉ hỗ trợ thêm Canxi-Bo.

---

### 3. Lưu ý quan trọng về thời gian giãn cách
* Cần giữ khoảng cách thời gian hợp lý giữa các lần bón (theo khuyến cáo của bài phân hoặc nhà sản xuất), **không bón quá thường xuyên, dồn dập**. Bón quá dày sẽ khiến rễ cây bị quá tải, dẫn đến cháy rễ và cháy lá.

---

**Một số câu hỏi để hỗ trợ bạn tốt hơn:**
1. Cây mai của bạn hiện là mai phôi mới ghép hay cây mai thuần/thành phẩm?
2. Cây của bạn hiện đang ở tháng thứ mấy hoặc đang trong giai đoạn phát triển nào (ra đọt non, nuôi cành lá hay đang tạo nụ)?

- **Đáp án tham chiếu (Ground Truth):**
> Với cây mai sau ghép, không nên bón phân sớm; tài liệu khuyên dùng phân hữu cơ sau ít nhất 3 tháng từ ngày tháo bọc, trừ khi phôi đã thuần chậu thì có thể ăn phân sớm hơn.

### Mẫu 6 [disease_nam_hong_01]: Dấu hiệu nhận biết bệnh nấm hồng là gì?
- **Phân loại intent:** `disease_lookup` | **Tags:** `diagnosis, leaf-symptom, pink-fungus`
- **Tổng thời gian xử lý:** `44.3485s`
- **Câu trả lời sinh ra:**
> Dấu hiệu nhận biết bệnh nấm hồng trên cây mai vàng qua các đặc điểm sau:

### 1. Dấu hiệu nhận biết (Triệu chứng)
* **Trên lá:** 
  * Lá thường bị vàng không đều, chủ yếu vàng lệch một bên mặt lá, phần còn lại ít bị hơn.
  * Bệnh chỉ xuất hiện tập trung ở các lá trên một cành hoặc nhánh cụ thể (chứ không bị vàng đồng loạt toàn cây như hiện tượng nóng rễ do phân thuốc).
  * Làm hư hại toàn bộ bộ lá trên chính chi cành bị nhiễm bệnh.
* **Trên thân và cành:**
  * Vỏ cây có hiện tượng bị nứt da, lột vỏ.
  * Xuất hiện tình trạng bỏ chi (chi cành bị nấm dần khô héo, chết cành và làm toàn bộ cây mai suy kiệt rất nhanh).

---

### 2. Nguyên nhân và điều kiện thuận lợi phát sinh bệnh
* Thời tiết ẩm ướt, mưa nhiều kéo dài làm nấm phát triển và lây lan nhanh.
* Cây trồng ở nơi thiếu thông thoáng, độ ẩm cao hoặc tàng lá quá um tùm, rậm rạp.
* Một số giống mai nhạy cảm, dễ mắc nấm hồng hơn như: mai vàng zin, các giống mai đột biến (dảo xoắn, dảo cam, dảo cúc cam...).

---

### 3. Hướng xử lý cơ bản
* **Cắt tỉa:** Nếu cành bị bệnh không ảnh hưởng đến dáng thế của cây, có thể cắt bỏ và đem đốt hoặc bỏ vào túi nilon vứt xa vườn để tiêu hủy mầm bệnh.
* **Lưu ý về lá:** Không nên lải bỏ lá bị bệnh (trừ đợt rong lá giữa năm hoặc lặt lá đón Tết), vì lá vẫn còn khả năng quang hợp nuôi cây.
* **Dùng thuốc xử lý:** Pha thuốc nấm (như COC85 hoặc Champion 77WP) quét trực tiếp lên vị trí thân/cành bị bệnh bằng cọ hoặc bàn chải. Khi vết bệnh đã khô/chết thì phun ngừa toàn thân bằng Anvil hoặc Ridomil Gold.

---

**Một số câu hỏi trao đổi thêm:**
1. Cây mai của bà con hiện tại chỉ mới bị vàng một vài nhánh lá hay thân cành đã có dấu hiệu nứt da, lột vỏ rồi?
2. Bà con có cần hướng dẫn chi tiết về liều lượng pha và cách quét thuốc đặc trị nấm hồng (COC85, Champion 77WP) cho từng mùa nắng/mưa không?

- **Đáp án tham chiếu (Ground Truth):**
> Tài liệu phân biệt nấm hồng với hiện tượng nóng rễ: nấm hồng thường làm vàng một bên lá, bên còn lại ít bị hơn.

### Mẫu 7 [disease_nam_re_01]: Bệnh nấm rễ gây hại cho cây mai ra sao?
- **Phân loại intent:** `disease_lookup` | **Tags:** `fungus, root-disease, symptoms`
- **Tổng thời gian xử lý:** `48.6337s`
- **Câu trả lời sinh ra:**
> Dựa vào thông tin trong tài liệu, bệnh nấm rễ gây hại rất nghiêm trọng cho cây mai vàng, cụ thể như sau:

### 1. Các tác hại cụ thể của nấm rễ lên cây mai
* **Làm suy yếu cây:** Nấm rễ chính là nguyên nhân phổ biến nhất dẫn đến tình trạng suy kiệt ở cây mai vàng.
* **Tấn công và phá hủy trực tiếp hệ rễ và gốc cây:**
  * Gây thối các đầu rễ non, rễ con (do nấm *Pythium*).
  * Gây thối toàn bộ rễ và làm xì mủ ở phần gốc cây (do nấm *Phytophthora*).
  * Gây tắc nghẽn các mạch dẫn dinh dưỡng và nước, khiến cây bị héo rũ (do nấm *Fusarium*).
  * Gây thối phần cổ rễ/thân nằm sát mặt đất (do nấm *Rhizoctonia*).
* **Diễn biến âm thầm và lây lan nhanh:** Bệnh nấm rễ phát triển rất nhanh nhưng lại không có biểu hiện rõ ràng trên bề mặt, rất khó quan sát bằng mắt thường. Thông thường, khi người trồng phát hiện ra thì bộ rễ đã hư hại nặng và cây đã rơi vào tình trạng suy yếu trầm trọng.

---

### 2. Nguyên nhân phát sinh chính
* Đất hoặc chất trồng bị đọng nước, úng ngập, thoát nước kém.
* Mùa mưa kéo dài kèm độ ẩm không khí và độ ẩm chất trồng quá cao.
* Chất trồng lâu năm bị bết, lèn chặt khiến đất bị bí và thiếu oxy cho rễ thở.

---

### 3. Hướng phòng ngừa và xử lý cơ bản
* **Phòng ngừa định kỳ:** Sử dụng thuốc nấm (như *Ridomil Gold*, *Trichoderma*...) tưới gốc định kỳ vào các thời điểm: sau khi thay/thêm chất trồng mới và vào các đợt trước, trong và sau mùa mưa.
* **Xử lý khi cây bị bệnh:** Phá bỏ chất trồng cũ, cạo/cắt bỏ hoàn toàn các đoạn rễ bị thối nhũn, sau đó ngâm/xử lý thuốc nấm (*Ridomil Gold*) kết hợp chất kích rễ (*Atonik*) theo phác đồ hướng dẫn để phục hồi lại rễ mới.

---

**Một số câu hỏi mở rộng:**
1. Cây mai của anh/chị hiện tại có biểu hiện bất thường nào như héo rũ, vàng lá hay chất trồng bị đọng nước lâu ngày không?
2. Anh/chị có muốn tìm hiểu chi tiết hơn về cách pha thuốc và quy trình ủ bọc nilon để cứu cây mai bị thối rễ nặng không?

- **Đáp án tham chiếu (Ground Truth):**
> Nấm rễ có thể làm hư rễ non và rễ cám, giảm khả năng hút dinh dưỡng. Khi biểu hiện thành lá vàng, rụng nhanh hoặc héo rũ thì tỷ lệ suy cây rất cao.

### Mẫu 8 [disease_rep_sap_01]: Rệp sáp gây hại cây mai như thế nào?
- **Phân loại intent:** `disease_lookup` | **Tags:** `mealybug, pest, symptoms`
- **Tổng thời gian xử lý:** `48.8314s`
- **Câu trả lời sinh ra:**
> Dựa vào dữ liệu, rệp sáp gây hại cho cây mai vàng theo các cách cụ thể sau:

**1. Tác hại trực tiếp lên cây:**
* **Chích hút dinh dưỡng:** Rệp sáp thường ẩn nấp ở các vị trí khuất ánh sáng như mặt dưới lá, cuống lá, kẽ lá, kẽ cành, thân cây và đặc biệt là ẩn sâu dưới bộ rễ (nhất là cây phôi, cây lâu ngày chưa thay chất trồng).
* **Làm suy kiệt cây:** Khi rệp chích hút nhựa, cây mai sẽ dần bị kiệt sức, xuất hiện triệu chứng vàng lá, kích thước hoa nở ra bị nhỏ lại và cây sẽ ngày càng suy yếu theo thời gian nếu không được can thiệp kịp thời.

**2. Tác hại gián tiếp (tạo điều kiện cho sâu hại và nấm bệnh khác):**
* **Thu hút và lây lan kiến:** Rệp sáp tiết ra chất dịch ngọt, đây là nguồn thức ăn ưa thích của kiến. Kiến sẽ bảo vệ rệp và tha rệp đi phát tán khắp các nhánh cây, khiến mật độ gây hại tăng nhanh.
* **Gây ra nấm bồ hóng:** Dịch ngọt do rệp sáp tiết ra còn tạo môi trường lý tưởng cho nấm bồ hóng sinh sôi, phủ một lớp đen trên bề mặt cây, làm cản trở quá trình quang hợp khiến cây càng thêm suy nhược.

---
**Một số câu hỏi gợi ý để hỗ trợ bà con tốt hơn:**
1. Cây mai của bà con hiện đang có biểu hiện rệp sáp ở vị trí nào (lá, thân nhánh hay dưới bộ rễ)?
2. Bà con có muốn tìm hiểu thêm về cách kiểm tra, phòng ngừa rệp sáp hoặc cách xử lý kết hợp ổ kiến quanh vườn không?

- **Đáp án tham chiếu (Ground Truth):**
> Rệp sáp chích hút làm cây suy yếu, lá vàng, bông nhỏ và cây yếu dần. Chúng thường xuất hiện ở mặt dưới lá, cuống lá, kẽ lá hoặc kẽ nhánh; chà vào vị trí nghi bệnh có dịch nhầy là dấu hiệu rệp.

### Mẫu 9 [disease_bo_tri_01]: Làm sao nhận biết bọ trĩ trên cây mai?
- **Phân loại intent:** `disease_lookup` | **Tags:** `leaf-symptom, pest, thrips`
- **Tổng thời gian xử lý:** `44.8528s`
- **Câu trả lời sinh ra:**
> Dựa trên thông tin được cung cấp, bà con có thể nhận biết bọ trĩ gây hại trên cây mai qua một số biểu hiện sau:

* **Trên lá non:** Khi lá non bị bọ trĩ chích hút, lá vẫn tiếp tục lớn lên nhưng sẽ có hiện tượng bị biến dạng, méo mó hoặc dị tật chứ không phát triển bình thường.
* **Trên nụ hoa:** Bọ trĩ thường xuất hiện và tấn công chích hút trực tiếp vào các nụ xanh (đặc biệt là giai đoạn cây bắt đầu ôm nụ xanh sau khi lặt lá đón Tết).

*(Lưu ý: Tài liệu chưa mô tả chi tiết hình thái của con bọ trĩ mà chỉ nêu các dấu hiệu nhận biết qua tác hại trên đọt non và nụ hoa).*

---
**Một số câu hỏi gợi ý để bà con cùng trao đổi thêm:**
1. Cây mai của bà con hiện tại đang ở giai đoạn ra đọt non hay đang ôm nụ xanh?
2. Bà con có cần tìm hiểu thêm về lịch xịt thuốc và cách phòng trừ bọ trĩ định kỳ cho cây không?

- **Đáp án tham chiếu (Ground Truth):**
> Khi cây ra mầm, chồi hoặc lá non, bọ trĩ chích hút có thể làm lá xoăn và biến dạng, ảnh hưởng trực tiếp đến sức sinh trưởng.

### Mẫu 10 [disease_dom_la_01]: Bệnh đốm lá trên mai có biểu hiện gì?
- **Phân loại intent:** `disease_lookup` | **Tags:** `leaf-spot, symptoms, fungus`
- **Tổng thời gian xử lý:** `48.3388s`
- **Câu trả lời sinh ra:**
> Chào bạn, về biểu hiện của bệnh đốm lá trên cây mai vàng, bạn có thể nhận biết qua các dấu hiệu cụ thể sau:

### 1. Dấu hiệu nhận biết (Biểu hiện)
* **Vết đốm đặc trưng:** Trên mặt lá xuất hiện những chấm tròn nhỏ li ti, xung quanh mỗi chấm có một quầng màu vàng bao bọc.
* **Vị trí và mức độ lây lan:** Bệnh thường bộc phát trước tiên trên các lá già với mật độ dày đặc, sau đó lây lan rất nhanh sang các lá non và chồi non của cây.

---

### 2. Nguyên nhân & Tác hại
* **Nguyên nhân:** Do nấm *Pestalozzia palmarum* gây ra, phát triển mạnh trong điều kiện thời tiết ẩm thấp, mưa nhiều hoặc cây có tàn lá quá rậm rạp, thiếu độ thông thoáng.
* **Tác hại:** Khiến cây mai suy giảm khả năng sinh trưởng và ảnh hưởng xấu đến việc ra hoa.

---

### 3. Hướng xử lý và Lưu ý
* **Thuốc điều trị:** Sử dụng các loại thuốc gốc đồng như Copper Oxychloride / Copper OxyClorua (chẳng hạn như thuốc Coc 85 pha theo tỉ lệ 10g cho 8 lít nước) hoặc dầu Neem (Neem oil), phun đều lên thân, lá và những khu vực đang bị bệnh.
* **Lưu ý quan trọng:** 
  * Phun thuốc nhằm mục đích chặn đứng mầm bệnh, không cho lây lan sang lá khác. Những lá đã bị đốm sẽ không thể tự hết vết đốm được.
  * Không nên vội vàng lặt bỏ hết các lá bị bệnh (trừ giai đoạn giữa năm hoặc sau Tết để làm thoáng cây), vì lá dù có đốm vẫn còn chức năng quang hợp tạo dưỡng chất nuôi cây.

---

**Một số câu hỏi mở rộng để bạn tham khảo thêm:**
1. Cây mai của bạn hiện đang bị đốm ở tầng lá già hay đã lan lên đọt non?
2. Bạn có muốn tìm hiểu thêm về cách phun phòng ngừa bệnh nấm lá định kỳ trong mùa mưa không?
3. Bạn có cần hướng dẫn chi tiết hơn về cách chăm sóc và xử lý tán lá để cây nhanh phục hồi không?

- **Đáp án tham chiếu (Ground Truth):**
> Biểu hiện dễ thấy là chấm tròn li ti có quầng vàng; chúng thường xuất hiện dày trên lá già rồi lan sang lá non và chồi non.

### Mẫu 11 [mixed_nam_hong_01]: Mai bị nấm hồng, cần xử lý và phòng tái phát thế nào?
- **Phân loại intent:** `mixed_reasoning` | **Tags:** `disease, pink-fungus, prevention`
- **Tổng thời gian xử lý:** `52.7324s`
- **Câu trả lời sinh ra:**
> Chào bạn, dưới đây là thông tin chi tiết và hướng dẫn cụ thể về cách nhận diện, xử lý và phòng ngừa bệnh nấm hồng trên cây mai theo tài liệu:

---

### 1. Dấu hiệu và sự nguy hiểm của bệnh nấm hồng
* **Triệu chứng & Tác hại:** Nấm hồng gây nứt da, lột vỏ, làm bỏ chi và hư toàn bộ lá trên nhánh bị bệnh. Biểu hiện vàng lá do nấm hồng thường chỉ bị một bên mặt lá (bên còn lại ít bị hơn). Nếu không xử lý kịp thời, cành nhánh sẽ bị khô, chết và khiến cây suy kiệt rất nhanh.
* **Yếu tố dễ phát sinh:** Thời tiết mưa nhiều, độ ẩm cao; cây có tàng nhánh quá rậm rạp, um tùm hoặc không thông thoáng. Ngoài ra, một số giống như mai dảo xoắn, dảo cam, dảo cúc cam hay mai vàng zin thường nhạy cảm và dễ mắc bệnh hơn.

---

### 2. Hướng xử lý cành và lá bị bệnh
* **Về lá:** Tuyệt đối không lải (vặt) bỏ lá bệnh chỉ vì thấy xấu, vì lá vẫn có khả năng quang hợp nuôi cây. Chỉ tiến hành tỉa/lải lá vào 2 dịp: rong lá giữa năm hoặc tuốt lá đón Tết.
* **Về cành nhánh:** Nếu việc cắt tỉa không làm ảnh hưởng đến dáng thế/tàng nhánh của cây thì có thể cắt bỏ cành bị bệnh. Cành cắt xong bắt buộc phải gom đem đốt (tiêu hủy) hoặc cho vào bao nilon buộc kín vứt xa vườn để tránh nấm lây lan.

---

### 3. Quy trình các bước trị bệnh nấm hồng
* **Bước 1 (Pha thuốc đậm đặc để quét):** 
  * Pha thuốc nấm **COC85** (liều lượng 10g với khoảng 50 - 100ml nước) hoặc dùng **CHAMPION 77WP**.
  * *Lưu ý:* Mùa mưa nên ưu tiên dùng COC85 vì độ bám dính cao hơn; mùa nắng nên dùng CHAMPION 77WP sẽ cho hiệu quả mạnh hơn.
* **Bước 2 (Quét thuốc):** Dùng cọ, bàn chải hoặc khăn vải quét trực tiếp nước thuốc đã pha lên vết nấm ở thân và cành.
* **Bước 3 (Kiểm tra lại):** Sau 5 đến 10 ngày, kiểm tra lại vị trí quét:
  * *Nếu nấm chưa chết/chưa cháy:* Tiếp tục pha thuốc quét lần 2.
  * *Nếu nấm đã cháy/chết:* Pha thuốc nấm như **RIDOMIL GOLD** hoặc **Anvil** để xịt đều lên toàn bộ thân, cành và lá.

---

### 4. Cách chăm sóc để phòng nấm hồng tái phát
* **Phun ngừa định kỳ:** Thường xuyên xịt ướt đều từ thân, cành, nhánh cho đến gốc bằng thuốc phòng nấm để hạn chế nấm quay trở lại.
* **Tăng cường cho cây nhạy cảm:** Những cây đã từng bị nấm hồng hoặc các giống mai nhạy cảm cần được lên lịch phòng ngừa thường xuyên hơn bình thường.
* **Tạo độ thông thoáng:** Tránh để cây mai ở nơi ẩm thấp, thiếu sáng; tỉa bớt các cành nhánh quá kín để vườn luôn thông thoáng.

---

**Một số câu hỏi mở rộng để hỗ trợ bạn tốt hơn:**
1. Cây mai của bạn hiện đang bị nấm hồng ở vị trí thân chính hay các cành nhánh phụ?
2. Giống mai bạn đang trồng là giống gì và khu vực trồng có đang vào mùa mưa hay ẩm ướt nhiều không?

- **Đáp án tham chiếu (Ground Truth):**
> Quét hỗn hợp thuốc nấm lên thân hoặc cành bị nấm hồng, kiểm tra sau 5–10 ngày và quét tiếp nếu chưa cháy/chết. Sau khi diệt được nấm, xịt thuốc nấm ướt thân, cành, gốc thường xuyên để phòng tái phát.

### Mẫu 12 [mixed_rep_sap_01]: Mai có rệp sáp thì xử lý thuốc gì và bao lâu kiểm tra lại?
- **Phân loại intent:** `mixed_reasoning` | **Tags:** `mealybug, monitoring, pest`
- **Tổng thời gian xử lý:** `51.0616s`
- **Câu trả lời sinh ra:**
> Chào bà con, đối với tình trạng rệp sáp gây hại trên cây mai vàng, việc phát hiện sớm và xử lý đúng cách là rất quan trọng để tránh cây bị vàng lá, suy yếu và ra hoa nhỏ.

Dưới đây là các loại thuốc điều trị và hướng dẫn theo dõi cụ thể từ tài liệu:

### 1. Các loại thuốc và dung dịch xử lý rệp sáp
Bà con có thể lựa chọn thuốc đặc trị hóa học hoặc các dung dịch tự nhiên tùy theo điều kiện thực tế:

* **Thuốc đặc trị hóa học:**
  - **MARSHAL 200SC:** Pha với tỷ lệ từ 1ml đến 2ml cho 1 lít nước.
  - **MOVENTO 150OD:** Pha đúng liều 1ml cho 1 lít nước.
  - **SHEBA 50EW:** Pha với tỷ lệ 2ml đến 3ml cho 1 lít nước.

* **Dung dịch sinh học / tự chế xua đuổi rệp:**
  - **Dung dịch Tỏi - Ớt - Gừng:** Giã nhuyễn 1 củ tỏi, 1 trái ớt cay và 1 lát gừng nhỏ, đem ngâm trong 1 lít nước qua đêm (hoặc đun sôi để nguội). Sau đó lọc bỏ bã, hòa thêm 1-2 muỗng cà phê nước rửa chén rồi xịt lên cây. Nếu dùng cách này, xịt định kỳ từ 3 đến 5 ngày một lần.
  - **Dầu Neem (Neem Oil):** Pha từ 5-10ml dầu Neem nguyên chất cùng 1 lít nước ấm và 1-2ml nước rửa chén hữu cơ.

---

### 2. Bao lâu cần kiểm tra lại?
* **Không cần xịt phòng ngừa định kỳ**, thay vào đó bà con cần **theo dõi thường xuyên** cây mai.
* Cần kiểm tra kỹ các vị trí khuất ánh sáng, mặt dưới lá, cuống lá, kẽ nhánh và đặc biệt là bộ rễ (nhất là mai phôi mới bứng hoặc cây lâu ngày chưa thay chất trồng).
* Bất cứ khi nào kiểm tra và phát hiện có rệp sáp trở lại, bà con phải tiến hành xịt thuốc xử lý ngay lập tức.

---

### 3. Lưu ý quan trọng đi kèm để rệp không tái phát
* **Phải diệt đồng thời cả kiến:** Kiến ăn chất dịch ngọt do rệp tiết ra và sẽ tha rệp phát tán khắp cây. Do đó, cần diệt sạch ổ kiến quanh vườn, xung quanh gốc hoặc tưới thuốc xử lý trong chất trồng.
* Giữ vườn thông thoáng, đủ ánh nắng để hạn chế môi trường ẩm ướt rệp sáp ưa thích.
* Diệt rệp sáp sớm cũng sẽ ngăn ngừa được nấm bồ hóng phát triển trên cây.

---

**Bà con có thể chia sẻ thêm một số thông tin để được hỗ trợ tốt hơn:**
1. Cây mai của bà con hiện đang bị rệp ở trên cành lá non hay ở dưới phần gốc, bộ rễ?
2. Xung quanh gốc mai hiện có thấy đàn kiến bò nhiều hay xuất hiện các mảng nấm bồ hóng màu đen chưa?

- **Đáp án tham chiếu (Ground Truth):**
> Tài liệu không khuyên xịt phòng rệp sáp định kỳ. Thay vào đó, theo dõi thường xuyên các vị trí khuất nắng, kẽ lá, kẽ nhánh và bộ rễ; phát hiện rệp thì xịt thuốc ngay.

### Mẫu 13 [mixed_ung_re_01]: Cây mai vàng lá, héo rũ do nấm rễ thì cứu thế nào?
- **Phân loại intent:** `mixed_reasoning` | **Tags:** `root-disease, root-rot, treatment`
- **Tổng thời gian xử lý:** `57.3095s`
- **Câu trả lời sinh ra:**
> Chào bạn, tình trạng cây mai bị vàng lá, héo rũ do nấm rễ là vấn đề nghiêm trọng cần được cứu chữa bài bản qua các bước cụ thể như sau:

---

### **1. NGUYÊN NHÂN & DẤU HIỆU**
- **Dấu hiệu:** Cây có biểu hiện vàng lá bất thường, lá bị héo rũ, đầu rễ bị thối nhũn, xì mủ hoặc hư hỏng.
- **Nguyên nhân:** Do các loại nấm bệnh hại tấn công vào hệ rễ cây. Bệnh thường phát sinh mạnh khi môi trường rễ bị ẩm thấp, tưới nước quá nhiều, chất trồng giữ nước lâu, bị lèn chặt thiếu oxy hoặc thoát nước kém.

---

### **2. HƯỚNG DẪN CÁCH XỬ LÝ VÀ CỨU CÂY CHI TIẾT**

#### **Bước 1: Vệ sinh rễ và loại bỏ chất trồng cũ**
- Nhổ cây mai ra khỏi chậu, tiến hành phá bỏ 100% bầu đất/chất trồng cũ (vì chất trồng cũ đã chứa mầm bệnh nấm hoặc bị hư hỏng).
- Dùng kéo/dao sạch cắt hoặc đục bỏ toàn bộ các đoạn rễ bị thối, đen hoặc hư hại.

#### **Bước 2: Ngâm thuốc trị nấm và phục hồi rễ**
- **Pha dung dịch:** 
  - Thuốc nấm **Ridomil Gold** (Metalaxyl + Mancozeb): 2 gram / 1 lít nước.
  - Thuốc kích thích **Atonik**: 1 ml / 2 lít nước.
  - *(Trường hợp có thêm tuyến trùng hại rễ, có thể pha thêm Tervigo liều 1 ml / 1 lít nước).*
- **Cách ngâm:** Ngâm toàn bộ bộ rễ vào dung dịch trên từ **8 đến 12 tiếng**. Nếu cây bị suy kiệt nặng, có thể dùng vải sạch thấm dung dịch này quấn lên toàn bộ thân cây để ngâm dưỡng.

#### **Bước 3: Trồng lại và trùm bọc nilon (Ủ cây nơi mát)**
- Trồng cây vào chậu với **chất trồng mới hoàn toàn** (đảm bảo chất trồng tơi xốp, khô thoáng và đáy chậu thoát nước thật tốt).
- Tưới thật đẫm hỗn hợp thuốc đã pha ở Bước 2 vào gốc và chất trồng.
- Dùng bọc nilon lớn trùm kín toàn bộ cây và chậu (bịt kín hơi).
- Đem cây đặt ở nơi râm mát, tránh ánh nắng hoàn toàn trong khoảng **15 đến 30 ngày** (tuyệt đối không tháo bọc, không tưới thêm bất kỳ loại nước hay phân thuốc nào trong thời gian này).

#### **Bước 4: Theo dõi xả gió và tháo bọc nilon**
- Sau 15–20 ngày (với chậu nhỏ dưới 40cm) hoặc 30 ngày (với chậu từ 50cm trở lên), kiểm tra mầm lá:
  - *Nếu cây ra mầm, chồi mới:* Pha lại dung dịch thuốc ở Bước 2 tưới 2 – 5 lít (tùy kích cỡ chậu) rồi tiếp tục bọc kín lại.
  - *Nếu sau 2 tháng không thấy cây đâm chồi:* Cây đã mất khả năng hồi phục.
- Khi lá non phát triển to cỡ khoảng 2 ngón tay: Dùng vật nhọn đục 1 đến 3 lỗ nhỏ (bằng đầu đũa) trên bọc nilon để cây tập quen không khí (xả gió).
- Sau khoảng 10 – 15 ngày xả gió, tiến hành tháo bỏ hoàn toàn bọc nilon vào thời điểm trời mát, chiều tối hoặc ngày mưa mát (nếu thấy có bọ trĩ thì xịt sương nhẹ để phòng ngừa).
- Tiếp tục tưới dung dịch thuốc trị nấm (ở Bước 2) định kỳ **15 ngày / 1 lần**, tưới liên tục 3 lần rồi ngưng hoàn toàn các loại phân thuốc khác cho đến khi cây ra cơi lá thứ 2.

#### **Bước 5: Giải độc và nuôi dưỡng rễ phục hồi**
- **Giải độc:** Dùng **Vitamin B12** (loại 1000mcg pha với 0.5 lít nước) phun đều thân lá và tưới gốc. Thực hiện liên tiếp 3 lần, mỗi lần cách nhau 3 ngày.
- **Kích rễ nhẹ:** Sau khi dùng B12, sử dụng thuốc dưỡng rễ nhẹ (như Terrasorb 4 Root hoặc Root2) với liều lượng bằng **50% so với liều thông thường**, tưới định kỳ 10 ngày / 1 lần (tưới 3 lần liên tục) để chờ cơi lá thứ 2 bung đều.

---

### **MỘT SỐ CÂU HỎI MỞ RỘNG:**
1. Cây mai của bạn hiện đang trồng trong chậu kích thước bao nhiêu và tình trạng vàng lá, héo rũ đã xuất hiện lâu chưa?
2. Chất trồng hiện tại trong chậu cây của bạn gồm những thành phần gì (xơ dừa, trấu sống, hay đất thịt)?
3. Bạn đã tiến hành xử lý cắt tỉa hay tưới loại phân thuốc nào vào gốc trước đó chưa?

- **Đáp án tham chiếu (Ground Truth):**
> Lá vàng bất thường và héo rũ là dấu hiệu nấm rễ nặng. Khi úng nước mới chớm, tạo đường thoát nước bằng cách xăm lỗ xuống đáy chậu; nếu rễ bị nấm nặng thì cần xử lý triệt để bằng cách nhổ cây, phá bầu, cắt rễ hư và thay chất trồng mới khô thoáng hơn.

### Mẫu 14 [mixed_tuoi_01]: Tưới quá nhiều có thể làm mai bị gì và khắc phục ra sao?
- **Phân loại intent:** `mixed_reasoning` | **Tags:** `root-rot, treatment, watering`
- **Tổng thời gian xử lý:** `51.4949s`
- **Câu trả lời sinh ra:**
> Chào bà con, việc tưới nước quá nhiều (dư nước/úng nước) là một lỗi rất phổ biến khiến cây mai bị tổn thương nghiêm trọng. Dưới đây là các tác hại cụ thể và cách xử lý, phòng ngừa dựa trên kỹ thuật chăm sóc mai:

---

### 1. TƯỚI QUÁ NHIỀU NƯỚC LÀM CÂY MAI BỊ GÌ?

* **Gây ngạt thở và thiếu oxy ở bộ rễ:** Khi tưới quá nhiều, chất trồng luôn trong tình trạng bão hòa nước, không khí không lọt vào được làm rễ cây bị "ngạt thở".
* **Làm úng rễ, thối rễ và thoái hóa rễ:** 
  * Tích tụ các chất độc làm rễ chuyển sang màu nâu đen, bị thối nhũn và bốc mùi hôi.
  * Hư hỏng hệ thống lông hút (bộ phận hút dinh dưỡng), làm rễ bị "chai", mất khả năng hút nước và thức ăn.
* **Lá héo rũ dù đất vẫn ướt:** Do rễ đã bị hư/chai nên không thể đưa nước lên lá, dẫn đến hiện tượng mất cân bằng: đất sũng nước nhưng lá mai vẫn bị héo rũ.
* **Suy cây và chết dần:** Cây không hấp thụ được dinh dưỡng mới, phải rút dần lượng nhựa dự trữ để nuôi thân lá, lâu ngày cây kiệt sức và có thể chết.
* **Tạo điều kiện cho nấm bệnh và tuyến trùng tấn công:** Môi trường ẩm ướt, thiếu oxy là điều kiện thuận lợi cho nấm hại rễ và tuyến trùng bùng phát mạnh trong khi rễ mai đang bị suy yếu.
* **Dễ gây dư phân:** Nếu tưới quá nhiều nước có pha phân hoặc tưới phân quá thường xuyên trong tháng cũng khiến cây bị quá tải và ngộ độc phân.

---

### 2. CÁCH KIỂM TRA VÀ KHẮC PHỤC

Để khắc phục và phòng ngừa tình trạng dư nước, bà con cần thực hiện các bước sau:

* **Kiểm tra độ ẩm chậu mai:** Dùng ngón tay thọc sâu xuống mặt chậu khoảng 1 ngón tay. Nếu sau 2–3 ngày tưới mà đất vẫn còn ướt sũng hoặc kiểm tra lỗ thoát nước dưới đáy chậu thấy ứ đọng nước, có sình/bùn thì cây đang bị dư nước nghiêm trọng.
* **Xử lý thoát nước ở đáy chậu:** Đảm bảo lỗ thoát nước dưới đáy luôn thông thoáng. Khi vào chậu, cần lót một lớp vật liệu thoáng khí ở dưới đáy để nước chảy ra dễ dàng, không bị lèn chặt hay đọng sình bùn.
* **Điều chỉnh chất trồng và kích thước chậu:** 
  * Tránh dùng chậu quá lớn so với cây hoặc chất trồng giữ ẩm quá lâu (như quá nhiều xơ dừa, cám dừa, nhiều đất).
  * Định kỳ hàng năm nên xăm và chêm thêm chất trồng mới (nếu không thay chậu) để đất không bị bết dính, chai cứng hay thiếu oxy.
* **Phòng ngừa nấm rễ:** Khi chất trồng quá ẩm dễ sinh nấm bệnh, bà con có thể dùng thuốc **Ridomil Gold** pha tưới vào chất trồng để ngăn ngừa và bảo vệ bộ rễ khỏi thối nhũn.
* **Kiểm soát phân bón:** Không tưới lượng nước phân quá nhiều vào gốc trong một lần và không tưới phân quá dày (tránh tưới 4–5 lần NPK/tháng).

---

**Bà con có thể chia sẻ thêm về tình trạng hiện tại của cây mai không:**
1. Chậu mai của bà con hiện tại tưới bao lâu thì khô và lá có đang bị héo rũ hay rụng lá không?
2. Chất trồng trong chậu của bà con chủ yếu là gì (xơ dừa, trấu hay đất) và lỗ thoát nước dưới đáy chậu có bị nghẹt không?

- **Đáp án tham chiếu (Ground Truth):**
> Tưới liên tục mà không kiểm soát độ ẩm có thể làm chất trồng luôn ướt và gây hư hoặc thúi rễ. Khi úng nước mới chớm, cần tăng thoáng khí và thoát nước, như xăm lỗ xuống đáy chậu để phá cấu trúc lèn chặt và tạo đường thoát nước.

### Mẫu 15 [mixed_sau_01]: Sâu ăn lá mai nên dùng thuốc thế nào cho đúng liều?
- **Phân loại intent:** `mixed_reasoning` | **Tags:** `dosage, leaf-eating-pest, pesticide`
- **Tổng thời gian xử lý:** `51.6172s`
- **Câu trả lời sinh ra:**
> Dưới đây là hướng dẫn cụ thể về cách sử dụng thuốc trị sâu ăn lá trên cây mai vàng theo đúng kỹ thuật:

---

### 1. Nguyên tắc dùng thuốc
* **Không cần xịt phòng liên tục:** Bà con không nhất thiết phải xịt ngừa định kỳ dày đặc, mà nên thường xuyên kiểm tra vườn (mỗi ngày hoặc 2–3 ngày/lần) ở các vị trí như đọt non, nụ và cành.
* **Dùng thuốc đặc trị khi phát hiện sâu:** Khi thấy có sâu xuất hiện, cần sử dụng ngay **thuốc trừ sâu đặc trị**. Tránh dùng thuốc bọ trĩ để diệt sâu vì sâu rất dễ kháng thuốc và không mang lại hiệu quả.

---

### 2. Kỹ thuật xịt và thời điểm xịt
* **Cách xịt:** Chỉ nên xịt một lớp sương nhẹ đều qua các tán lá, không xịt quá dày hay xịt đẫm làm ảnh hưởng xấu đến lá.
* **Thời điểm lý tưởng:** Nên xịt vào lúc trời mát mẻ, tốt nhất là vào **buổi chiều tối**. Tuyệt đối không xịt thuốc vào thời điểm trời đang nắng nóng.
* **Tần suất (khi kết hợp quản lý sâu và bọ trĩ):**
  * Vườn ít cây: Cách khoảng **10 – 15 ngày/lần**.
  * Vườn nhiều cây: Cách khoảng **7 – 10 ngày/lần**.

---

### 3. Cách điều chỉnh liều lượng theo điều kiện thời tiết (Mùa mưa)
* **Xịt vào sáng sớm (khi lá còn ướt nước mưa):** Pha nồng độ thuốc đậm hơn một chút (tăng thêm khoảng **20% – 30%** liều lượng) hoặc pha thêm chất bám dính để bù đắp lượng nước đọng trên lá.
* **Xịt vào giữa buổi sáng (09 – 10 giờ) hoặc đầu giờ chiều (15 – 16 giờ):** Pha loãng thuốc (chỉ dùng khoảng **60% – 70%** liều lượng thông thường), xịt sương nhẹ để tránh làm "nóng" và cháy lá non.

---

### Câu hỏi mở rộng để hỗ trợ bạn tốt hơn:
1. Cây mai của bạn hiện đang là mai phôi mới ghép hay cây mai đã thành phẩm?
2. Ngoài lá non bị sâu ăn, cây có đang xuất hiện hiện tượng bị bọ trĩ hay nấm bệnh nào khác không?

- **Đáp án tham chiếu (Ground Truth):**
> Sâu ăn lá cần thuốc đặc trị phù hợp. Tài liệu liệt kê, ví dụ, Karate 2.5EC với liều điều trị 1–1,5 ml/1 lít nước và Delfin 32WG với liều 1 gram/1 lít nước; nên đối chiếu đúng thuốc trước khi pha.

### Mẫu 16 [general_ghep_01]: Bấm ngọn mai dùng để làm gì?
- **Phân loại intent:** `general_lookup` | **Tags:** `branching, pruning, training`
- **Tổng thời gian xử lý:** `47.2907s`
- **Câu trả lời sinh ra:**
> Dựa vào thông tin trong tài liệu, việc **bấm chặn (bấm ngọn/bấm đọt)** trên cây mai vàng có các tác dụng chính sau:

1. **Giúp tàn nhánh dày đặc:** Đối với mai tàng nuôi tự nhiên, việc bấm ngọn thường xuyên (cứ ra khoảng 3 - 4 lá thì bấm) sẽ kích thích cây phân nhiều nhánh, làm cho bộ tàn trở nên kín và dày đặc.
2. **Tạo và giữ dáng (form) cây:** Giúp cây duy trì được khung tàn tròn đều, cân đối để chơi cây quanh năm mà không nhất thiết phải dùng kẽm uốn.
3. **Định hình và giúp cây phát triển cân đối:** Đối với mai ghép, bấm ngọn kết hợp tỉa cành giúp cây phát triển khỏe mạnh, tạo tán hài hòa; đồng thời kiểm soát những cành phát triển quá nhanh để chừa lại các nhánh đi đúng hướng, thuận tiện cho việc vô kẽm định hình sau này.

---

**Một số câu hỏi mở rộng để trao đổi thêm cùng bạn:**
- Cây mai của bạn hiện là mai ghép hay mai tàng nuôi tự nhiên (mai nguyên thủy)?
- Bạn đang muốn bấm tỉa để tạo dáng tàn thông thoáng đón mưa hay chuẩn bị cho giai đoạn làm nụ cuối năm?

- **Đáp án tham chiếu (Ground Truth):**
> Trong tài liệu, thao tác tương ứng được gọi là cắt đôn hoặc bấm đọt: cắt chặn ngọn, cành vượt hoặc cành sai hướng để cây phân nhánh nhiều hơn và dồn dinh dưỡng vào các cành còn lại.

### Mẫu 17 [general_dat_01]: Chất trồng cho mai cần có đặc điểm gì?
- **Phân loại intent:** `general_lookup` | **Tags:** `drainage, growing-medium, soil`
- **Tổng thời gian xử lý:** `61.6269s`
- **Câu trả lời sinh ra:**
> Dựa trên tài liệu, để bộ rễ mai phát triển khỏe mạnh và cây sinh trưởng tốt, chất trồng cho cây mai vàng cần đảm bảo các đặc điểm kỹ thuật sau:

1. **Thoát nước nhanh và tốt:** Cây mai là loại cây cần nước nhưng ưa khô ráo và rất sợ ngập úng. Do đó, chất trồng phải thoát nước cực kỳ tốt, không giữ độ ẩm quá lâu để tránh gây ứ đọng nước hoặc sinh ra bùn/sình ở đáy chậu làm thối rễ.
2. **Thông thoáng và tơi xốp:** Đất trồng không được nén hay lèn chặt mà cần có độ tơi xốp cao để cung cấp nhiều oxy cho bộ rễ hô hấp. Tránh dùng chất trồng đã quá cũ bị phân hủy mục nát lâu ngày (nếu cũ cần phải thay mới).
3. **Được bổ sung nấm có lợi phòng bệnh:** Trước khi sử dụng, chất trồng nên được phối trộn với nấm đối kháng *(Trichoderma)* để ức chế các loại nấm bệnh gây hại cho rễ.

---

**Một số câu hỏi mở rộng để hỗ trợ bác/anh/chị tốt hơn:**
1. Cây mai của bác/anh/chị hiện tại đang được trồng bằng loại chất trồng nào (ví dụ: xơ dừa, tro trấu, đất mặt ruộng...)?
2. Cây có đang gặp hiện tượng vàng lá hay ứ nước ở đáy chậu cần tư vấn xử lý không?

- **Đáp án tham chiếu (Ground Truth):**
> Chất trồng phù hợp cần thoát nước tốt, không giữ ẩm quá lâu và tránh làm chậu úng nước; đồng thời nên tơi xốp, nhiều oxy, không nén/lèn chặt.

### Mẫu 18 [general_tuoi_01]: Vì sao không nên để nước cũ trong chất trồng?
- **Phân loại intent:** `general_lookup` | **Tags:** `growing-medium, root-care, watering`
- **Tổng thời gian xử lý:** `47.2964s`
- **Câu trả lời sinh ra:**
> Việc để nước ứ đọng hoặc chất trồng bị dư nước trong thời gian dài sẽ gây ra nhiều tác hại nghiêm trọng cho cây mai vàng, cụ thể như sau:

---

### **1. Tác hại và nguyên nhân khi chất trồng bị ứ đọng nước**
* **Làm ngạt rễ, thiếu oxy:** Khi nước chiếm hết các khoảng trống trong chất trồng, không khí không thể lọt vào, khiến rễ cây bị thiếu oxy nghiêm trọng (ngạt thở).
* **Gây thúi rễ, hư lông hút:** Do bị ngạt và tích tụ độc chất, rễ sẽ chuyển sang màu nâu đen, nhũn ra và bốc mùi hôi. Các đầu lông hút (bộ phận lấy dinh dưỡng và nước) bị thoái hóa hoặc bị "chai", khiến cây mất khả năng hấp thu dinh dưỡng.
* **Cây bị suy kiệt, héo lá:** Khi bộ rễ hư hỏng không hút được nước, cây sẽ xảy ra hiện tượng mất cân bằng (dù đất ướt nhưng lá vẫn héo rũ), cây phải tiêu hao hết lượng nhựa dự trữ dẫn đến suy yếu dần và có thể chết.
* **Tạo điều kiện cho nấm bệnh bùng phát:** Môi trường ẩm ướt, bí khí là nơi lý tưởng cho các loại nấm hại và tuyến trùng phát triển mạnh mẽ để tấn công bộ rễ đang suy yếu.

---

### **2. Dấu hiệu nhận biết**
* Rễ chuyển màu nâu đen, mềm nhũn, có mùi hôi.
* Chất trồng luôn trong tình trạng sũng nước, bết dính.
* Cây chậm phát triển, lá bị héo rũ dù đất bên dưới vẫn ướt.

---

### **3. Hướng xử lý khi cây bị dư nước, đọng nước (theo tài liệu)**
* **Trường hợp mới chớm bị úng:** Dùng một que thép nhỏ (khoảng 4–5 ly) xăm nhiều lỗ thông từ trên mặt xuống tận đáy chậu để phá vỡ lớp đất bị lèn chặt, giúp nước thoát nhanh và tạo khoảng thở cho rễ. Nếu chậu quá to so với bầu rễ, nên moi bớt lớp chất trồng thừa ở các góc chậu.
* **Trường hợp bị úng nặng, chất trồng vón cục, sình bùn:** Cần xử lý triệt để bằng cách nhổ cây lên, loại bỏ 100% bầu đất cũ, cắt bỏ hết phần rễ bị thối/hư, sau đó thay hoàn toàn bằng chất trồng mới khô thoáng, thoát nước tốt hơn.
* **Chăm sóc phục hồi:** Đặt cây ở nơi râm mát, thoáng gió (như dưới tán cây, mái hiên, lưới lan) để tránh cháy rễ và sốc nhiệt. Có thể tưới thêm thuốc trị nấm rễ (như Ridomil Gold) vào chất trồng để ngăn ngừa và xử lý nấm bệnh tấn công.

---

**Một số câu hỏi mở rộng để hỗ trợ bà con tốt hơn:**
1. Cây mai của bà con hiện tại có đang gặp tình trạng lá héo rũ hoặc chất trồng bị sũng nước không?
2. Chậu trồng hiện tại của bà con là loại chậu nào (chậu thường hay chậu ly sâu lòng) và đáy chậu có lỗ thoát nước tốt không?

- **Đáp án tham chiếu (Ground Truth):**
> Tài liệu hướng dẫn tưới gốc thật đẫm sau khoảng 10–30 ngày, tùy chậu và độ khô của chất trồng, để làm mới nước trong chất trồng và tránh chất trồng khô tầng dưới đáy.

### Mẫu 19 [general_mai_tu_quy_01]: Mai Tứ Quý tự lột vỏ có bình thường không?
- **Phân loại intent:** `general_lookup` | **Tags:** `bark, mai-tu-quy, natural-growth`
- **Tổng thời gian xử lý:** `49.7735s`
- **Câu trả lời sinh ra:**
> Hiện tượng **mai Tứ Quý tự lột vỏ là hoàn toàn bình thường và tự nhiên**. 

Đây là một đặc tính sinh trưởng sinh học tự nhiên của giống mai Tứ Quý để giúp thân cây phát triển và lớn lên. Điểm này khác với cây mai vàng truyền thống, vì mai vàng rất hiếm khi tự lột vỏ/da thân nhiều như mai Tứ Quý. 

Do đó, nếu bà con thấy cây mai Tứ Quý của mình tự bong tróc lớp vỏ ngoài để lộ lớp vỏ mới bên trong trong quá trình sinh trưởng thì có thể yên tâm, không phải là dấu hiệu của bệnh hại.

---

**Một số câu hỏi mở rộng để hỗ trợ thêm cho bà con:**
1. Cây mai của bà con là mai Tứ Quý thuần hay đang dùng gốc Tứ Quý để ghép mai vàng?
2. Vùng da bị lột có thấy lớp vỏ non bên dưới hay bị lộ phần lõi gỗ bên trong kèm theo dấu hiệu nấm bệnh, nứt nẻ nào khác không?

- **Đáp án tham chiếu (Ground Truth):**
> Có. Tài liệu nêu mai Tứ Quý có khả năng tự lột vỏ tự nhiên để lớn lên; mai vàng thì ít lột vỏ hơn.

### Mẫu 20 [general_insufficient_01]: Có thuốc nào trị được mọi bệnh trên cây mai không?
- **Phân loại intent:** `general_lookup` | **Tags:** `fungicide, insufficient-evidence, scope`
- **Tổng thời gian xử lý:** `54.089s`
- **Câu trả lời sinh ra:**
> Dạ theo thông tin trong tài liệu, **không có một loại thuốc duy nhất nào có thể trị được tất cả mọi bệnh** trên cây mai vàng.

Để chăm sóc và trị bệnh cho cây mai hiệu quả, bà con cần lưu ý một số điểm sau:

1. **Nên luân phiên nhiều loại thuốc:** Bà con nên chuẩn bị sẵn vài loại thuốc nấm trong vườn để thay đổi qua lại khi xịt. Việc này giúp hạn chế tình trạng nấm bị lờn thuốc và phát huy tối đa hiệu quả phòng trừ.
2. **Bộ thuốc nấm tham khảo:** Bộ ba thuốc được đề xuất sử dụng nhiều là **Ridomil Gold, Anvil và Coc85**. Sự kết hợp luân phiên này rất tiện dụng và có thể xử lý được **đa số** các bệnh nấm thường gặp trên cây mai vàng (nhưng không phải toàn bộ mọi bệnh).
3. **Lưu ý về bệnh trên lá:** Đối với các bệnh hại lá (như nấm, bọ trĩ, nhện đỏ...), việc xịt thuốc chỉ có tác dụng tiêu diệt mầm bệnh để chặn lây lan và không cho bệnh nặng thêm, chứ các vết tích trên lá (vàng lá, cháy bìa, đốm đen) sẽ không biến mất được cho đến khi cây thay cơi lá mới. Vì vậy, việc xịt thuốc phòng ngừa định kỳ luôn quan trọng hơn đợi cây có bệnh mới trị.

---
**Một số câu hỏi để em có thể hỗ trợ bà con tốt hơn:**
- Cây mai của bà con hiện đang gặp phải dấu hiệu cụ thể nào (ví dụ: đốm lá, xoắn/cong lá hay bị vấn đề ở rễ)?
- Gần đây bà con đã sử dụng qua loại phân bón hay thuốc xịt nào cho cây chưa?

- **Đáp án tham chiếu (Ground Truth):**
> Knowledge base không thiết lập một thuốc trị mọi bệnh. Nó chỉ nêu một combo Ridomil Gold, Anvil và Coc85 là tiện dụng, hiệu quả và trị được đa số bệnh; vì vậy không nên suy diễn đây là thuốc hay phác đồ cho mọi bệnh.
