"""
Task 7 — Reranking Module.

PHƯƠNG PHÁP ĐÃ CHỌN: RRF (Reciprocal Rank Fusion)

VÌ SAO CHỌN RRF thay vì Cross-encoder hay MMR?
    - Không cần API key / model tải về: Cross-encoder (Jina Reranker) cần JINA_API_KEY
      và gọi mạng; RRF chỉ là phép tính thuần trên rank, chạy local tức thì.
    - Đúng bài toán Hybrid Search của pipeline này: Task 9 cần gộp kết quả từ Task 5
      (semantic/dense) và Task 6 (BM25/sparse) — hai thang điểm này KHÔNG cùng đơn vị
      (cosine similarity 0-1 vs BM25 score không giới hạn trên), nên không thể cộng
      trực tiếp. RRF giải quyết đúng vấn đề này bằng cách chỉ dùng THỨ HẠNG (rank) của
      mỗi kết quả trong từng danh sách, bỏ qua thang điểm gốc — nhờ vậy công bằng khi
      gộp nhiều ranker có thang điểm khác nhau.
    - MMR giải quyết vấn đề khác (đa dạng hoá kết quả, giảm trùng lặp), không phải vấn
      đề gộp nhiều ranker, nên không phù hợp mục tiêu Task 7/9 ở đây.

CÔNG THỨC: RRF(d) = Σ 1 / (k + rank_r(d)), k=60 (hằng số smoothing, theo paper gốc
Cormack et al. 2009 — giá trị 60 được chọn thực nghiệm để giảm ảnh hưởng của các rank
thấp/nhiễu mà không cần tinh chỉnh theo từng dataset).

CƠ CHẾ: với mỗi ranked list (ví dụ kết quả từ semantic_search, kết quả từ lexical_search),
duyệt từng document theo thứ hạng 1, 2, 3... rồi cộng dồn điểm 1/(k+rank) vào tổng điểm
của document đó (dùng content làm khoá để nhận diện document trùng giữa các list). Cuối
cùng sắp xếp theo tổng điểm giảm dần. Document nào xuất hiện ở thứ hạng cao trong NHIỀU
danh sách sẽ có điểm tổng cao nhất — đúng tinh thần "đồng thuận" giữa các ranker.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó (Task 9 phải dùng điểm Cosine gốc từ
Task 5, chưa qua RRF, để quyết định ngưỡng fallback < 0.48).
"""

from typing import Optional


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    # TODO: Implement cross-encoder reranking
    #
    # Option A: Jina Reranker API
    # import requests
    # response = requests.post(
    #     "https://api.jina.ai/v1/rerank",
    #     headers={"Authorization": f"Bearer {JINA_API_KEY}"},
    #     json={
    #         "model": "jina-reranker-v2-base-multilingual",
    #         "query": query,
    #         "documents": [c["content"] for c in candidates],
    #         "top_n": top_k
    #     }
    # )
    # reranked = response.json()["results"]
    # return [
    #     {**candidates[r["index"]], "score": r["relevance_score"]}
    #     for r in reranked
    # ]
    #
    # Option B: Local model (Qwen3-Reranker)
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # ...
    raise NotImplementedError("Implement rerank_cross_encoder")


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    # TODO: Implement MMR
    #
    # selected = []
    # remaining = list(range(len(candidates)))
    #
    # for _ in range(min(top_k, len(candidates))):
    #     best_idx = None
    #     best_score = float('-inf')
    #
    #     for idx in remaining:
    #         # Relevance to query
    #         relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
    #
    #         # Max similarity to already selected
    #         max_sim_to_selected = 0
    #         for sel_idx in selected:
    #             sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
    #             max_sim_to_selected = max(max_sim_to_selected, sim)
    #
    #         # MMR score
    #         mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
    #
    #         if mmr_score > best_score:
    #             best_score = mmr_score
    #             best_idx = idx
    #
    #     selected.append(best_idx)
    #     remaining.remove(best_idx)
    #
    # return [candidates[i] for i in selected]
    raise NotImplementedError("Implement rerank_mmr")


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}   # content -> tổng điểm RRF
    content_map: dict[str, dict] = {}   # content -> item gốc (giữ metadata)

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Giữ lại item đầu tiên gặp — content giống nhau thì metadata cũng giống nhau
            content_map.setdefault(key, item)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = dict(content_map[content])  # copy để không sửa item gốc trong ranked_lists
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict] | list[list[dict]],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval. Với method="rrf" có thể truyền:
            - 1 list đơn (List[dict]) — vd kết quả từ 1 nguồn duy nhất, RRF sẽ áp dụng
              trên chính thứ tự đó (tương đương "chuẩn hoá điểm về thang rank").
            - nhiều list (List[List[dict]]) — vd [semantic_results, bm25_results] ở
              Task 9, RRF sẽ gộp (fuse) thứ hạng từ tất cả các list.
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước rồi gọi rerank_mmr() trực tiếp
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # Tự nhận diện: list các dict (1 nguồn) hay list các list (nhiều nguồn cần fuse)
        if candidates and isinstance(candidates[0], list):
            ranked_lists = candidates
        else:
            ranked_lists = [candidates]
        return rerank_rrf(ranked_lists, top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
