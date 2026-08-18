# Failure Analysis — Lab 18: Production RAG

**Sinh viên:** Hoàng Đức Anh
**MSSV:** 2A202601223
**Module:** M1 → M5 (cá nhân, implement toàn bộ)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8333 | 0.8423 | +0.0090 |
| Answer Relevancy | 0.7513 | 0.7703 | +0.0190 |
| Context Precision | 0.9375 | 0.9250 | -0.0125 |
| Context Recall | 0.9167 | 0.8000 | **-0.1167** |

**Nhận xét nhanh:** Production cải thiện faithfulness và answer_relevancy nhờ rerank + enrichment (context sạch hơn, LLM bám sát nguồn hơn), nhưng **context_recall giảm mạnh (-11.67%)**. Nguyên nhân chính (xem #3 bên dưới): hierarchical child chunks (256 ký tự) cắt các bảng markdown (VD: bảng thẩm quyền phê duyệt) thành nhiều mảnh nhỏ, khiến một số hàng dữ liệu quan trọng (VD: mức phê duyệt CEO) không nằm trong top-3 context được rerank, trong khi naive dùng basic paragraph chunking (500 ký tự) giữ được nhiều bảng trọn vẹn hơn trong 1 chunk.

## Bottom-5 Failures

### #1
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải thay đổi mỗi 120 ngày. Chính sách cũ (90 ngày) đã bị thay thế.
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness (0.3958)
- **Error Tree:** Output sai (từ chối trả lời) → Context đúng? **Có** — cả 2 chunk (v1: 90 ngày, v2: 120 ngày + note "đã được thay thế bởi v2.0") đều nằm trong top-3 → Query OK? **Có**, câu hỏi rõ ràng → **Fix ở bước generation**: LLM thấy 2 con số mâu thuẫn (90 vs 120) trong context, không có rule nào hướng dẫn cách resolve version conflict → chọn an toàn là từ chối thay vì suy luận "note superseded → dùng số mới nhất".
- **Root cause:** System prompt không có hướng dẫn xử lý tài liệu nhiều phiên bản (versioning conflict). Đây là loại câu hỏi "version" trong test_set (đúng như thiết kế 6 category của assignment) — điểm yếu thực sự nằm ở prompt engineering, không phải retrieval.
- **Suggested fix:** Thêm rule vào system prompt: "Nếu có nhiều phiên bản tài liệu mâu thuẫn, ưu tiên bản có ghi chú 'hiện hành'/ngày hiệu lực gần nhất, bỏ qua bản có note 'đã thay thế'." Hoặc lọc metadata theo `version`/`status` trước khi đưa vào context.

### #2
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30tr nằm trong khoảng 5-50tr → Director phê duyệt; cần xác nhận cấu hình kỹ thuật từ CNTT; đính kèm ≥3 báo giá (vì >10tr).
- **Got:** "Không tìm thấy."
- **Worst metric:** answer_relevancy (0.5208)
- **Error Tree:** Output sai → Context đúng? **Không** — top-3 context chỉ có chunk "xác nhận cấu hình CNTT" + 2 chunk không liên quan (tạm ứng, đào tạo); **thiếu hẳn** chunk bảng "Thẩm quyền phê duyệt theo giá trị đơn hàng" (5-50tr → Director) → Query OK? Có, nhưng câu hỏi dùng từ "laptop 30 triệu" trong khi chunk bảng chỉ có số liệu dạng "5.000.000 VNĐ – 50.000.000 VNĐ | Director" không có từ "laptop" → **Fix ở bước retrieval**.
- **Root cause:** Mismatch từ vựng — không có overlap lexical giữa "laptop 30 triệu" và bảng khoảng giá trị (không có BM25 match), và dense embedding cũng không rank bảng số này đủ cao vì câu hỏi thiên về "loại thiết bị" còn chunk thiên về "khoảng giá trị". Cần suy luận số học (30 nằm trong khoảng 5-50) mà embedding không nắm bắt tốt.
- **Suggested fix:** Tăng `HYBRID_TOP_K`/`RERANK_TOP_K` cho nhóm câu hỏi numeric, hoặc dùng query rewriting (LLM diễn giải lại "laptop 30 triệu" → "thiết bị giá trị 30.000.000 VNĐ") trước khi search để tăng lexical/semantic overlap với bảng.

### #3 *(Case Study — xem chi tiết bên dưới)*
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng >50.000.000 VNĐ → Tổng Giám đốc (CEO) phê duyệt.
- **Got:** "Cần có phê duyệt của Kế toán trưởng." **(SAI)**
- **Worst metric:** context_recall (0.7014)
- **Error Tree:** Output sai → Context đúng? **Không** — chunk retrieve được chỉ có 2 dòng đầu bảng ("Dưới 5tr → Trưởng phòng | 5-50tr → ...") rồi **bị cắt cụt** do `child_size=256`, thiếu hẳn dòng "Trên 50tr → CEO" → Query OK? Có → **Fix ở bước chunking (M1)**.
- **Root cause:** Hierarchical chunking cắt theo `child_size` cố định (256 ký tự) không nhận biết ranh giới bảng markdown, làm vỡ bảng "Thẩm quyền phê duyệt" thành nhiều child không đầy đủ. LLM sau đó lẫn lộn với chunk khác nói về "Kế toán trưởng" (vốn thuộc quy định tạm ứng, không phải mua sắm) → hallucinate câu trả lời sai hoàn toàn.
- **Suggested fix:** Dùng `chunk_structure_aware` (đã implement ở M1) cho các tài liệu có bảng thay vì `chunk_hierarchical` theo char-count thuần, hoặc thêm rule "không cắt giữa 1 bảng markdown" khi chia child chunks.

### #4
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày (20-15 ngày thanh toán) × phí 2%/tháng trên 15.000.000đ ≈ pro-rata ~50.000đ cho 5 ngày.
- **Got:** "Bị phạt 2% của 15 triệu VNĐ. Số tiền phạt là 300.000 VNĐ."
- **Worst metric:** faithfulness (0.7868)
- **Error Tree:** Output sai (thiếu bước pro-rate) → Context đúng? **Có** — context có đủ 2 fact cần thiết (hạn 15 ngày, phí 2%/tháng trên số tiền chưa hoàn ứng) → Query OK? Có → **Fix ở bước generation (multi-hop numeric reasoning)**.
- **Root cause:** LLM tính đúng công thức 2% × 15tr = 300k nhưng **bỏ qua bước pro-rate theo số ngày trễ thực tế** (5/30 ngày) mà câu hỏi yêu cầu — trả lời cả tháng phí thay vì phí tương ứng 5 ngày trễ. Đây là lỗi reasoning nhiều bước (multi-hop numeric), đúng category "numeric" trong test_set.
- **Suggested fix:** Thêm few-shot example tính phí pro-rata vào system prompt, hoặc tăng lý luận từng bước bằng chain-of-thought prompt ("Tính số ngày trễ trước, sau đó tính phí tương ứng").

### #5
- **Question:** Mentor và buddy của nhân viên mới có thể là cùng một người không? Quản lý trực tiếp có thể làm mentor không?
- **Expected:** KHÔNG cho cả hai — Mentor và Buddy phải là 2 người khác nhau; quản lý trực tiếp không được làm mentor/buddy.
- **Got:** "Mentor và buddy của nhân viên mới không thể là cùng một người. Quản lý trực tiếp không được làm Mentor hoặc Buddy." **(ĐÚNG, khớp ground truth)**
- **Worst metric:** faithfulness (0.7946)
- **Error Tree:** Output đúng về nội dung → nhưng RAGAS vẫn chấm faithfulness thấp vì claim-decomposition coi câu trả lời diễn giải ("không thể") là suy luận rộng hơn câu gốc trong context ("phải là hai người khác nhau") thay vì trích dẫn nguyên văn.
- **Root cause:** Đây là **false positive của metric**, không phải lỗi hệ thống thực sự — minh chứng RAGAS faithfulness nhạy với cách paraphrase, không chỉ đúng/sai ngữ nghĩa.
- **Suggested fix:** Không cần sửa hệ thống; nếu muốn tăng điểm faithfulness, có thể chỉnh prompt để LLM trích dẫn gần với văn phong nguồn hơn, nhưng đánh đổi là câu trả lời kém tự nhiên hơn — không đáng làm.

## Case Study (cho presentation)

**Question chọn phân tích:** #3 — "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"

**Error Tree walkthrough:**
1. Output đúng? → **Sai** — trả lời "Kế toán trưởng" thay vì "CEO/Tổng Giám đốc".
2. Context đúng? → **Sai** — chunk retrieve được là nửa đầu bảng phê duyệt (chỉ có "Dưới 5tr" và phần đầu "5-50tr"), **thiếu dòng "Trên 50tr → CEO"**. Bảng gốc trong `mua_sam.md` có 3 dòng nhưng `chunk_hierarchical` với `child_size=256` chỉ giữ được ~2 dòng đầu trong 1 child.
3. Query rewrite OK? → Có, câu hỏi rõ ràng, không cần rewrite.
4. Fix ở bước: **M1 — Chunking**. Đây không phải lỗi search/rerank/generation, mà là lỗi ở tận gốc: dữ liệu cần thiết đã bị chunking cắt rời trước khi kịp vào index.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Chuyển các tài liệu chứa bảng (mua_sam.md, bang_luong_2024.md, tam_ung.md...) sang `chunk_structure_aware` thay vì `chunk_hierarchical`, hoặc tăng `child_size` đủ lớn để 1 bảng không bị vỡ.
- Thêm bước "table-aware splitting": không cắt chunk giữa markdown table (`|...|` liên tiếp).
- Thêm rule version-conflict vào system prompt (fix cho #1) và few-shot pro-rata reasoning (fix cho #4) — 2 chỗ này không tốn thêm compute, chỉ cần sửa prompt.
