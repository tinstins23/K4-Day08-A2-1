"""
Task 4 — Chunking & Indexing vào Vector Store.

Pipeline: load markdown (data/standardized/) → chunk → embed → index vào ChromaDB.

Cài đặt:
    pip install langchain-text-splitters chromadb google-genai python-dotenv

Chạy:
    python src/task4_chunking_indexing.py

Lưu ý quan trọng: nếu sau này đổi corpus hoặc đổi embedding provider, phải XOÁ
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection (và dimension khác nhau sẽ gây lỗi), retrieval sẽ trả về rác.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# Corpus: legal dài (điều khoản ~350–530 ký tự) + news ngắn (~600–1600 ký tự)
CHUNK_SIZE = 800        # ~1–2 điều khoản / 1 mục help; đủ ngữ cảnh, chưa trộn nhiều chủ đề
CHUNK_OVERLAP = 100     # ~12%; tránh cắt giữa câu pháp lý dài, giữ mạch sang chunk kế
# -----------------------------------------------------------------------------
# A. CHUNKING STRATEGY: MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
# -----------------------------------------------------------------------------
# VÌ SAO KẾT HỢP 2 SPLITTER (chiến lược 2 tầng)?
#
#   Tầng 1 — MarkdownHeaderTextSplitter (chia theo CẤU TRÚC):
#     Toàn bộ corpus sau Task 3 đều là Markdown và có heading rõ ràng
#     (ví dụ "## [Trả hàng/Hoàn tiền] ...", "### 1. Khiếu nại chưa nhận được hàng",
#     "1. ĐỐI TƯỢNG VÀ PHẠM VI"). Cắt theo heading giữ cho mỗi chunk nằm trọn trong
#     MỘT chủ đề, không bị trộn nội dung "chính sách hoàn tiền" với "cách theo dõi
#     đơn hàng". Quan trọng hơn: splitter này bơm heading vào metadata (h1/h2/h3),
#     nên sau này ở Task 5/7 ta biết chunk thuộc mục nào của tài liệu để hiển thị
#     nguồn trích dẫn cho người dùng.
#
#   Tầng 2 — RecursiveCharacterTextSplitter (chia theo KÍCH THƯỚC):
#     Chỉ dùng tầng 1 là không đủ. Các file legal (chính sách Shopee, 27-103KB) có
#     những section dài hàng chục nghìn ký tự dưới cùng một heading — chunk như vậy
#     vượt xa context hữu ích và làm loãng embedding (một vector 768 chiều không thể
#     biểu diễn tốt 20.000 ký tự). Tầng 2 cắt tiếp các section quá dài theo ranh giới
#     tự nhiên (đoạn văn → dòng → câu → từ), đảm bảo mọi chunk đều <= CHUNK_SIZE.
#
#   Tóm lại: tầng 1 lo NGỮ NGHĨA (chunk đúng chủ đề), tầng 2 lo KÍCH THƯỚC (chunk
#   đủ nhỏ để embed chính xác). Dùng riêng lẻ thì thiếu một trong hai.
CHUNKING_METHOD = "markdown_header + recursive"

# VÌ SAO CHUNK_SIZE = 500?
#   - Corpus là văn bản hướng dẫn/chính sách tiếng Việt: một ý trọn vẹn (một bước
#     thao tác, một điều khoản) thường gói trong 300-500 ký tự. 500 đủ chứa trọn ý
#     mà không kéo theo nhiều ý lạc đề.
#   - Tiếng Việt có dấu tốn token hơn tiếng Anh (~1 token/2-3 ký tự), 500 ký tự
#     ≈ 170-250 token — vừa vặn để nhồi top-5 chunk vào prompt Task 10 mà không
#     tràn context.
#   - Chunk quá lớn (1000+) làm embedding bị "trung bình hoá", giảm độ chính xác
#     retrieval; chunk quá nhỏ (<200) làm mất ngữ cảnh, câu trả lời bị cụt.


# Heading nào sẽ được tách ở tầng 1 (và tên metadata tương ứng)
MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


# -----------------------------------------------------------------------------
# B. EMBEDDING MODEL: Google Gemini text-embedding-004
# -----------------------------------------------------------------------------
# VÌ SAO CHỌN GEMINI text-embedding-004?
#   1. Đa ngôn ngữ & mạnh với tiếng Việt: corpus là tiếng Việt có dấu, thuật ngữ
#      TMĐT ("mã vận đơn", "Trả hàng/Hoàn tiền", "ShopeePay"). Model này được train
#      đa ngôn ngữ nên biểu diễn tiếng Việt tốt hơn các model chỉ mạnh tiếng Anh
#      như all-MiniLM-L6-v2.
#   2. Nhẹ về cài đặt: gọi qua API nên không phải tải torch + model weights (~1-2GB
#      như BAAI/bge-m3 hay sentence-transformers). Máy yếu / máy không có GPU vẫn
#      chạy được, thời gian setup ngắn.
#   3. Miễn phí ở mức dùng của bài lab: Google AI Studio cấp free tier đủ để embed
#      vài trăm chunk, không cần thẻ tín dụng như OpenAI.
#   4. Có task_type chuyên biệt (RETRIEVAL_DOCUMENT khi index, RETRIEVAL_QUERY khi
#      truy vấn) — đây là lợi thế thật sự: model sinh vector khác nhau cho document
#      và cho query, giúp điểm cosine ở Task 5 chính xác hơn so với dùng chung 1 vector.
#   Đánh đổi: cần GEMINI_API_KEY và cần mạng; bù lại nhẹ và chất lượng tiếng Việt tốt.
#
#   Lưu ý (cập nhật): model text-embedding-004 đã bị Google deprecate (14/1/2026).
#   Model hiện hành là gemini-embedding-001, dùng Matryoshka Representation Learning
#   (MRL) nên có thể cắt bớt vector 3072 chiều gốc xuống 768 (output_dimensionality=768)
#   mà không phải train lại — giữ được ưu điểm nhẹ/nhanh như lựa chọn ban đầu.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google")
EMBEDDING_MODEL = "gemini-embedding-001"

# VÌ SAO DIMENSION = 768?
#   gemini-embedding-001 mặc định trả 3072 chiều, nhưng hỗ trợ cắt (MRL) xuống 768/1536.
#   768 vẫn là lựa chọn cân bằng: nhẹ hơn 3072 tới 4 lần (ChromaDB tìm kiếm nhanh hơn,
#   tốn ít đĩa hơn) trong khi chất lượng retrieval trên corpus nhỏ (vài trăm chunk)
#   gần như không khác biệt so với dùng full 3072 chiều.
EMBEDDING_DIM = 768

# Batch size khi gọi API embed — tránh gửi quá nhiều text 1 lần gây timeout/429
# (free tier Gemini giới hạn 100 "đơn vị"/phút cho embed_content, mỗi item trong batch
# tính là 1 đơn vị — batch 20 + nghỉ giữa các batch giữ tốc độ an toàn dưới ngưỡng đó)
EMBED_BATCH_SIZE = 20
EMBED_SLEEP_SECONDS = 8.0


# -----------------------------------------------------------------------------
# C. VECTOR STORE: ChromaDB
# -----------------------------------------------------------------------------
# VÌ SAO CHỌN CHROMADB?
#   1. Local & persistent, KHÔNG cần Docker: chỉ cần PersistentClient(path=...) là
#      có DB lưu ở thư mục chroma_db/. Weaviate mạnh hơn nhưng phải dựng Docker/Cloud,
#      quá nặng cho phạm vi bài lab.
#   2. Lưu kèm metadata và cho phép filter (where={"type": "legal"}) — cần thiết cho
#      Task 5/9 khi muốn giới hạn tìm kiếm theo loại tài liệu.
#   3. Hơn FAISS ở chỗ FAISS chỉ là index vector thuần, phải tự quản lý mapping
#      id → text → metadata; ChromaDB gói sẵn cả ba.
#   4. Hỗ trợ cosine similarity qua metadata {"hnsw:space": "cosine"} — đúng độ đo
#      cần cho dense retrieval ở Task 5.
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "ecommerce_support_docs"

# Dùng cosine vì embedding đã chuẩn hoá hướng ngữ nghĩa; cosine đo góc giữa 2 vector
# nên không bị ảnh hưởng bởi độ dài văn bản (chunk dài/ngắn vẫn so sánh công bằng).
DISTANCE_METRIC = "cosine"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo chiến lược 2 tầng:
        Tầng 1: MarkdownHeaderTextSplitter — cắt theo heading (giữ ngữ nghĩa)
        Tầng 2: RecursiveCharacterTextSplitter — cắt tiếp phần quá dài (giữ kích thước)

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # giữ lại dòng heading trong nội dung chunk để không mất ngữ cảnh
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []

    for doc in documents:
        # --- Tầng 1: cắt theo heading ---
        try:
            header_sections = header_splitter.split_text(doc["content"])
        except Exception:
            header_sections = []

        # File không có heading (ví dụ PDF convert ra text phẳng) → fallback nguyên văn
        if not header_sections:
            sections = [(doc["content"], {})]
        else:
            sections = [(s.page_content, dict(s.metadata or {})) for s in header_sections]

        # --- Tầng 2: cắt tiếp theo kích thước ---
        chunk_index = 0
        for section_text, header_meta in sections:
            if not section_text.strip():
                continue

            for piece in char_splitter.split_text(section_text):
                if not piece.strip():
                    continue

                meta = {**doc["metadata"], "chunk_index": chunk_index}
                # Ghi heading vào metadata (ChromaDB chỉ nhận scalar: str/int/float/bool)
                for key in ("h1", "h2", "h3"):
                    if header_meta.get(key):
                        meta[key] = str(header_meta[key])
                # Tiêu đề gần nhất — tiện hiển thị nguồn trích dẫn ở Task 10
                meta["section"] = str(
                    header_meta.get("h3")
                    or header_meta.get("h2")
                    or header_meta.get("h1")
                    or doc["metadata"].get("source", "")
                )

                chunks.append({"content": piece, "metadata": meta})
                chunk_index += 1

    return chunks


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Embed danh sách text bằng provider đã cấu hình.

    Tách riêng hàm này để Task 5 gọi lại được với task_type="RETRIEVAL_QUERY",
    tránh viết logic embed lặp lại ở 2 nơi.
    """
    if EMBEDDING_PROVIDER != "google":
        raise ValueError(
            f"File này cấu hình cho provider 'google', đang nhận '{EMBEDDING_PROVIDER}'. "
            "Sửa EMBEDDING_PROVIDER=google trong .env."
        )

    # Dùng SDK mới `google-genai` (package `google.generativeai` cũ đã bị Google khai
    # tử và không tương thích với định dạng API key mới "AQ.xxx" qua REST endpoint cũ
    # — sẽ báo lỗi 404 "model not found" dù model/key đều đúng).
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu GEMINI_API_KEY. Mở file .env và dán key vào dòng GEMINI_API_KEY=... "
            "(lấy key miễn phí tại https://aistudio.google.com/app/apikey)"
        )
    client = genai.Client(api_key=api_key)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        result = _embed_batch_with_retry(client, types, batch, task_type)
        vectors.extend([e.values for e in result.embeddings])
        print(f"  ... embedded {min(start + EMBED_BATCH_SIZE, len(texts))}/{len(texts)}")

        # Giãn nhịp giữa các batch để không dồn dập chạm rate limit free tier
        # (embed_content_free_tier_requests: 100 đơn vị/phút — mỗi item trong batch
        # tính là 1 đơn vị, nên batch 50 gần như chạm trần chỉ sau 2 request liên tiếp).
        if start + EMBED_BATCH_SIZE < len(texts):
            time.sleep(EMBED_SLEEP_SECONDS)

    return vectors


def _embed_batch_with_retry(client, types, batch: list[str], task_type: str, max_retries: int = 6):
    """
    Gọi embed_content với retry + exponential backoff khi gặp 429 (RESOURCE_EXHAUSTED).

    Free tier Gemini giới hạn số request/phút — 429 là chuyện bình thường khi embed
    hàng trăm chunk, KHÔNG phải lỗi key hay model sai. Ưu tiên dùng retryDelay mà
    Google trả về trong response error nếu có, nếu không thì backoff theo cấp số nhân.
    """
    from google.genai import errors as genai_errors

    delay = 10.0
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
        except genai_errors.ClientError as e:
            is_rate_limit = getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
            if not is_rate_limit or attempt == max_retries:
                raise

            # Google thường trả sẵn thời gian nên chờ trong 'retryDelay' — ưu tiên dùng nó
            suggested = _extract_retry_delay_seconds(e)
            wait_s = suggested if suggested else delay
            print(f"  ⚠ 429 rate limit (free tier), thử lại sau {wait_s:.0f}s "
                  f"(lần {attempt}/{max_retries})...")
            time.sleep(wait_s)
            delay = min(delay * 2, 120.0)


def _extract_retry_delay_seconds(error) -> float | None:
    """Đọc 'retryDelay' (vd '7s') từ chi tiết lỗi 429 của Google, nếu có."""
    try:
        details = error.details or {}
        for item in details.get("error", {}).get("details", []):
            if item.get("@type", "").endswith("RetryInfo"):
                raw = item.get("retryDelay", "")
                return float(raw.rstrip("s")) + 1.0  # +1s đệm an toàn
    except Exception:
        pass
    return None


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    texts = [c["content"] for c in chunks]
    vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào ChromaDB (persistent, cosine similarity)."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_METRIC},
    )

    ids = [
        f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"  Collection '{COLLECTION_NAME}' hiện có {collection.count()} chunks")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking     : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding    : {EMBEDDING_MODEL} (dim={EMBEDDING_DIM}, provider={EMBEDDING_PROVIDER})")
    print(f"  Vector Store : {VECTOR_STORE} ({DISTANCE_METRIC})")
    print("=" * 60)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")
    if chunks:
        sizes = [len(c["content"]) for c in chunks]
        print(f"  Chunk size: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")

    print("\nEmbedding...")
    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")
    print(f"\n✓ Done! Vector store tại: {CHROMA_DIR}")


if __name__ == "__main__":
    run_pipeline()
