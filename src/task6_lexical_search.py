"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

STANDARDIZED_DIR = (
    Path(__file__).parent.parent
    / "data"
    / "standardized"
)

#CORPUS = []
BM25_INDEX = None
# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

def load_corpus():
    corpus = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        corpus.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": md_file.parent.name
            }
        })

    return corpus
def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # TODO: Implement BM25 index
    #
    # from rank_bm25 import BM25Okapi
    #
    # # Tokenize - có thể đơn giản split(), hoặc dùng underthesea cho tiếng Việt
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25
    #raise NotImplementedError("Implement build_bm25_index")


def lexical_search(query: str, top_k: int = 10) -> list[dict]:

    global CORPUS
    global BM25_INDEX

    # Chỉ load và build index lần đầu
    if BM25_INDEX is None:
        CORPUS = load_corpus()
        BM25_INDEX = build_bm25_index(CORPUS)

    tokenized_query = query.lower().split()

    scores = BM25_INDEX.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        if scores[idx] <= 0:
            continue

        results.append({
            "content": CORPUS[idx]["content"],
            "score": float(scores[idx]),
            "metadata": CORPUS[idx]["metadata"]
        })

    return results

if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
