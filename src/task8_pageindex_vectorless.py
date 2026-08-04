"""
Task 8 — PageIndex Vectorless RAG
"""

import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from pageindex.client import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

# lưu doc_id sau khi upload
DOC_ID = None


def upload_documents():
    """
    Upload PDF lên PageIndex.

    Returns:
        doc_id
    """
    global DOC_ID

    pdfs = list(LEGAL_DIR.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(
            f"Không tìm thấy PDF trong {LEGAL_DIR}"
        )

    pdf_path = pdfs[0]

    print(f"Uploading: {pdf_path.name}")

    resp = client.submit_document(str(pdf_path))

    print(json.dumps(resp, indent=2, ensure_ascii=False))

    DOC_ID = resp.get("doc_id")

    if not DOC_ID:
        raise RuntimeError("Upload thất bại")

    print("Waiting for indexing...")

    while not client.is_retrieval_ready(DOC_ID):
        print("  indexing...")
        time.sleep(5)

    print("✓ Ready")

    return DOC_ID


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.
    """

    global DOC_ID

    if DOC_ID is None:
        raise RuntimeError(
            "Chưa upload document. Hãy gọi upload_documents() trước."
        )

    submit = client.submit_query(
        doc_id=DOC_ID,
        query=query
    )

    retrieval_id = submit["retrieval_id"]

    retrieval = client.get_retrieval(retrieval_id)

    results = []

    nodes = retrieval.get("retrieved_nodes", [])

    rank = 0

    for node in nodes:

        for group in node.get("relevant_contents", []):

            for item in group:

                rank += 1

                results.append(
                    {
                        "content": item.get(
                            "relevant_content",
                            ""
                        ),
                        "score": round(1.0 - rank * 0.05, 3),
                        "metadata": {
                            "section": item.get(
                                "section_title",
                                ""
                            )
                        },
                        "source": "pageindex",
                    }
                )

                if len(results) >= top_k:
                    return results

    return results


if __name__ == "__main__":

    if not PAGEINDEX_API_KEY:
        print("Thiếu PAGEINDEX_API_KEY")
        exit()

    upload_documents()

    print()

    results = pageindex_search(
        "danh sách sản phẩm cấm đăng bán",
        top_k=5,
    )

    print()

    if not results:
        print("Không có kết quả")
    else:
        for r in results:
            print("=" * 60)
            print(r["score"])
            print(r["metadata"])
            print(r["content"][:300])