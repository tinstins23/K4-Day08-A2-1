"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from .task4_chunking_indexing import (
    embed_texts,
    CHROMA_DIR,
    COLLECTION_NAME,
)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }

        Sorted theo score giảm dần.
    """

    import chromadb

    # ============================================================
    # Bước 1: Embed query
    # ============================================================

    query_vector = embed_texts(
        [query],
        task_type="RETRIEVAL_QUERY",
    )[0]

    # ============================================================
    # Bước 2: Mở ChromaDB
    # ============================================================

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_collection(
        COLLECTION_NAME
    )

    # ============================================================
    # Bước 3: Query
    # ============================================================

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    # ============================================================
    # Bước 4: Convert output
    # ============================================================

    output = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        # cosine distance -> similarity
        score = max(0.0, 1.0 - distance)

        output.append(
            {
                "content": doc,
                "score": round(score, 4),
                "metadata": meta,
            }
        )

    output.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return output[:top_k]


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    queries = [
        "quy định trả hàng hoàn tiền shopee",
        "đổi phương thức thanh toán",
        "mã vận đơn",
        "hàng quốc tế",
    ]

    for q in queries:

        print("=" * 80)
        print("QUERY:", q)
        print("=" * 80)

        results = semantic_search(
            q,
            top_k=5,
        )

        for i, r in enumerate(results, 1):

            source = r["metadata"].get(
                "source",
                "Unknown",
            )

            print(f"\n[{i}] Score: {r['score']:.4f}")
            print("Source:", source)
            print(r["content"][:200], "...")