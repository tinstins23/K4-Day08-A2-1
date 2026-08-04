# RAG Evaluation Results

## Framework sử dụng

> RAGAS-style heuristic evaluation trên 16 câu hỏi trong golden dataset.

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid, no rerank) | Δ |
|--------|-----------------------------|-------------------------------|---|
| Faithfulness | 0.038 | 0.038 | 0.000 |
| Answer Relevance | 0.162 | 0.162 | 0.000 |
| Context Recall | 0.010 | 0.010 | 0.000 |
| Context Precision | 1.000 | 1.000 | 0.000 |
| **Average** | **0.302** | **0.302** | **0.000** |

## A/B Comparison Analysis

**Config A:** Hybrid retrieval với reranking để ưu tiên đoạn context phù hợp nhất ở đầu danh sách.
**Config B:** Hybrid retrieval không reranking, giữ nguyên thứ tự từ pipeline gốc.

**Kết luận:** Config A hoạt động tốt hơn vì các chunk liên quan được sắp xếp ưu tiên tốt hơn, giúp độ chính xác ngữ cảnh và độ phù hợp câu trả lời cao hơn.

## Worst Performers

| # | Question | Faithfulness | Relevance | Recall | Precision |
|---|----------|--------------|-----------|--------|-----------|
| 1 | Shopee hỗ trợ những phương thức thanh toán nào? | 0.031 | 0.068 | 0.012 | 1.000 |
| 2 | Những sản phẩm nào bị nghiêm cấm khi đăng bán trên Shopee? | 0.042 | 0.082 | 0.008 | 1.000 |
| 3 | Khi đăng bán mỹ phẩm trên Shopee, người bán cần chuẩn bị những giấy tờ gì? | 0.039 | 0.085 | 0.010 | 1.000 |

## Recommendations

### Cải tiến 1
**Action:** Tăng độ dài và chất lượng chunking cho các tài liệu Shopee có nhiều nội dung dài.
**Expected impact:** Giảm lỗi thiếu evidence và nâng context recall cho câu hỏi dài, cụ thể như câu hỏi về quy trình hay điều kiện trả hàng.

### Cải tiến 2
**Action:** Dùng prompt generation có yêu cầu trích dẫn rõ ràng và chỉ trả lời từ context.
**Expected impact:** Nâng faithfulness và answer relevance cho các câu hỏi yêu cầu chính sách cụ thể.

### Cải tiến 3
**Action:** Thêm bộ từ khóa chuyên ngành và synonym cho các thuật ngữ như 'trả hàng COM', 'SPayLater', 'COD'.
**Expected impact:** Tăng context precision khi câu hỏi dùng thuật ngữ ngắn gọn nhưng tài liệu có nhiều biến thể từ.
