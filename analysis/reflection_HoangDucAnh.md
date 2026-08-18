# Individual Reflection — Lab 18: Production RAG Pipeline

**Tên:** Hoàng Đức Anh
**MSSV:** 2A202601223
**Module phụ trách:** M1 → M5 (bài cá nhân, implement toàn bộ 5 module)

---

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 tạo **208 chunks** (avg 99 ký tự, min 6, max 354) so với basic paragraph chunking chỉ 51 chunks (avg 410 ký tự). Semantic chunking nhạy với threshold — nhiều câu ngắn không liên quan về topic bị tách thành chunk riêng (min=6 ký tự là một câu gần như trống), cho thấy threshold 0.85 hơi cao cho văn bản tiếng Việt ngắn câu như policy doc. |
| Parent-child hierarchical | M1 | `chunk_hierarchical()` | 99 child chunks (avg 210 ký tự, cap 256 theo `HIERARCHICAL_CHILD_SIZE`). Đây là strategy được chọn cho pipeline production vì cân bằng precision (child nhỏ → embedding chính xác) và context (parent lớn hơn). Nhưng thực tế index chỉ dùng child, không return parent — nên vẫn mất context nếu 1 bảng bị cắt giữa 2 child (xem failure #3 trong `failure_analysis.md`). |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 chunks theo markdown headers, giữ nguyên bảng/section — tốt hơn hierarchical cho tài liệu có bảng, nhưng pipeline hiện tại chưa dùng chunk này cho production (chỉ dùng hierarchical) → đây là action item rõ ràng cho action plan bên dưới. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | RRF giải quyết vấn đề BM25 và Dense "lệch pha" — BM25 match tốt câu hỏi có từ khóa chính xác ("nghỉ phép"), Dense match tốt câu hỏi diễn giải khác từ (paraphrase). Chìa khóa là `segment_vietnamese()` dùng underthesea rồi `replace("_", " ")` — nếu bỏ bước replace này, BM25 tokenize theo `split(" ")` sẽ coi "nghỉ_phép" là 1 token duy nhất, không khớp query "nghỉ phép" (2 token) — lỗi rất dễ mắc và khó debug vì không có exception, chỉ là kết quả search rỗng. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Benchmark thực tế: **avg 2126ms / min 260ms / max 9520ms** trên 5 docs. Chênh lệch lớn giữa min và max không phải do model chậm mà do **cold-start load model** (2.27GB, XLM-RoBERTa) chiếm gần hết thời gian — sau khi cache model ở module-level (thêm vào code so với scaffold gốc), các lần rerank tiếp theo trong cùng process chỉ mất vài chục ms. Đây là insight quan trọng cho production: phải warm-up reranker model lúc khởi động service, không load lazy trên request đầu tiên. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Context Precision cao nhất (0.9250) nhờ rerank lọc noise tốt; Answer Relevancy thấp nhất (0.7703) — cho thấy retrieval "sạch" nhưng generation đôi khi lạc trọng tâm câu hỏi ghép (multi-hop). Context Recall giảm so với naive (0.80 vs 0.9167) — bài học lớn nhất: **chunking nhỏ hơn không tự động tốt hơn**, vì có thể cắt vỡ thông tin liên kết (bảng, danh sách) cần thiết cho câu trả lời. |
| Contextual embeddings (Anthropic-style) | M5 | `contextual_prepend()`, `_enrich_single_call()` | Dùng combined mode (1 API call/chunk thay vì 4) cho 104 chunks, mất 285.5s tổng (~2.7s/chunk). Context prepend thực tế sinh ra, VD: *"Đoạn văn nằm trong phần hướng dẫn về bảo mật mật khẩu trong tài liệu mat_khau_v1.md"* — câu này giúp reranker/embedding hiểu chunk thuộc version nào, nhưng KHÔNG giải quyết được vấn đề version-conflict ở tầng generation (xem failure #1) vì contextual prepend chỉ mô tả vị trí, không tự resolve mâu thuẫn nội dung giữa 2 version. |

## Phần 2: Khó khăn & giải quyết

### Khó khăn 1 — Windows console UnicodeEncodeError
- **Lỗi:** `UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-3` khi chạy `naive_baseline.py`, do `print()` chứa emoji `⚠️` + tiếng Việt trong khi console Windows mặc định dùng `cp1252`.
- **Debug:** So sánh traceback trỏ thẳng vào dòng `print()` trong `load_documents()` → xác định nguyên nhân là console encoding, không phải lỗi logic.
- **Fix:** Chạy script với `PYTHONUTF8=1` để ép Python dùng UTF-8 mode, không cần sửa code.
- **Thời gian debug:** ~5 phút.

### Khó khăn 2 — Reranker crash liên tục (Segmentation fault / access violation)
- **Lỗi:** `Windows fatal exception: access violation` mỗi lần load `BAAI/bge-reranker-v2-m3` (2.27GB) qua `sentence_transformers.CrossEncoder`, kể cả khi chỉ gọi raw `safetensors.torch.load_file()`.
- **Debug (systematic, ~1.5 giờ):** Thử lần lượt: downgrade `transformers` (4.46.3 vs 5.15.0), downgrade `safetensors` (0.4.5 vs 0.8.0), test load model nhỏ hơn (`ms-marco-MiniLM`, load OK) để cô lập vấn đề về model cụ thể, test raw `safetensors.safe_open()` (cũng crash), parse header thủ công (OK, file không hỏng) → cuối cùng kiểm tra `Get-PSDrive C` phát hiện **ổ C: chỉ còn 205MB trống** và **RAM chỉ còn 2.45GB free / 15.6GB** (do nhiều app khác đang chạy: `pyrefly` 2.1GB, Docker WSL, nhiều IDE process).
- **Cách giải quyết:** Dọn `uv cache clean` (giải phóng 2.7GB), sau đó chuyển hẳn cache HuggingFace (~4.4GB) và cache `uv` sang ổ D: (`HF_HOME`, `UV_CACHE_DIR` set persistent qua `setx`) — ổ C tăng từ 205MB lên ~9.5GB trống. Root cause thực sự không phải bug thư viện mà là **tài nguyên hệ thống cạn kiệt** khiến mmap/malloc cho tensor lớn thất bại ngẫu nhiên.
- **Kiến thức thiếu → bổ sung:** Trước đây chưa từng nghĩ tới việc disk gần đầy có thể gây crash ở tầng ứng dụng Python (không liên quan trực tiếp I/O) — giờ hiểu rằng Windows cần disk headroom cho page file/memory-mapped file handling, không chỉ RAM thuần.
- **Thời gian debug:** ~90 phút.

### Khó khăn 3 — Mỗi test load lại model 2.27GB → crash khi chạy cả suite
- **Lỗi:** `test_rerank_returns` pass riêng lẻ nhưng `test_rerank_type` (chạy ngay sau) crash — vì mỗi `CrossEncoderReranker()` instance tự load model riêng, RAM không kịp giải phóng giữa 2 lần load trong cùng 1 process pytest.
- **Fix:** Thêm module-level cache `_model_cache: dict[str, object]` (theo đúng pattern đã có sẵn ở `m1_chunking.py._get_semantic_model()`), để nhiều instance `CrossEncoderReranker` dùng chung 1 model đã load thay vì load lại.
- **Thời gian debug:** ~10 phút (nhờ nhận ra pattern đã có sẵn trong code base).

## Phần 3: Action Plan cho project

```markdown
## Project: RAG nội bộ cho tài liệu chính sách/quy trình (mở rộng từ lab)

### Hiện tại
- RAG pipeline hiện tại: hierarchical chunking (child 256 ký tự) + BM25/Dense hybrid (RRF) + CrossEncoder rerank + enrichment 1-call/chunk + RAGAS eval.
- Known issues:
  1. Chunking cắt vỡ bảng markdown → mất thông tin (context_recall giảm 11.67% so với naive) — failure #3.
  2. Không có rule xử lý version-conflict giữa nhiều bản tài liệu — failure #1.
  3. Reranker cold-start ~9.5s nếu không warm-up trước.

### Plan áp dụng
1. [ ] Chunking strategy: chuyển tài liệu có bảng/danh sách sang `chunk_structure_aware` thay vì `chunk_hierarchical` thuần theo char-count; giữ hierarchical cho văn bản prose thông thường — chọn strategy theo loại tài liệu, không dùng 1 strategy cho tất cả.
2. [ ] Search: giữ Hybrid (BM25 + Dense + RRF) — đã chứng minh cải thiện answer_relevancy/faithfulness so với dense-only của naive baseline.
3. [ ] Reranking: giữ CrossEncoder `bge-reranker-v2-m3`, nhưng **warm-up model lúc service khởi động** (gọi `rerank()` 1 lần dummy) thay vì lazy-load ở request đầu, tránh timeout 9.5s cho user đầu tiên.
4. [ ] Evaluation: RAGAS cho 4 metrics tổng quan + `failure_analysis()` Diagnostic Tree để review nhanh bottom-N mỗi lần deploy; bổ sung theo dõi riêng metric `context_recall` vì đây là điểm yếu đã phát hiện.
5. [ ] Enrichment: giữ combined single-call mode (tiết kiệm chi phí 4x so với 4 call riêng), nhưng thêm rule metadata `version`/`status` vào `extract_metadata()` để giải quyết version-conflict ở bước retrieval/prompt thay vì để LLM tự đoán.

### Timeline
- Tuần 1: Refactor chunking để chọn strategy theo loại tài liệu (table-aware vs prose), viết test riêng cho case bảng bị cắt.
- Tuần 2: Thêm version-conflict metadata filtering + warm-up reranker vào service startup; đo lại RAGAS trước/sau, mục tiêu context_recall ≥ naive baseline (0.9167).
```
