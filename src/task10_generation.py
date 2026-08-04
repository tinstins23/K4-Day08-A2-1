"""
Task 10 — Generation Có Citation.

Pipeline:
    Retrieval (Task 9)
        ↓
    Document Reordering (anti lost-in-the-middle)
        ↓
    Context Formatting
        ↓
    LLM Generation
        ↓
    Answer + Citation

Yêu cầu:
    - Citation nguồn
    - Không hallucination
    - Nếu thiếu evidence → trả về thông báo không xác minh được
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================

# Số lượng chunk đưa vào LLM.
#
# Chọn top_k = 5:
# - Đủ context để trả lời câu hỏi phức tạp.
# - Không quá nhiều tránh:
#       + Context quá dài
#       + LLM bị "lost in the middle"
#       + Tăng chi phí token
TOP_K = 5


# top_p:
#
# Chọn 0.9:
# - Giữ câu trả lời tự nhiên.
# - Không quá cao để tránh sinh nội dung ngoài context.
TOP_P = 0.9


# Temperature:
#
# RAG ưu tiên tính chính xác hơn sáng tạo.
# Chọn 0.3 để giảm hallucination.
TEMPERATURE = 0.3


# Model sử dụng qua OpenRouter/OpenAI compatible API
#
# Có thể đổi sang model :free nếu không có credit.
LLM_MODEL = "openai/gpt-4o-mini"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """
Bạn là trợ lý AI hỗ trợ khách hàng thương mại điện tử.

Nhiệm vụ:
- Trả lời câu hỏi dựa ONLY trên context được cung cấp.
- Không được tự suy luận hoặc thêm thông tin không có trong context.

Quy tắc bắt buộc:

1. Mỗi thông tin quan trọng phải có citation ngay sau câu.
   Ví dụ:
   "Shopee cho phép thay đổi phương thức thanh toán trong một số trường hợp
   [Shopee Payment Policy, 2026]."

2. Citation phải lấy từ nguồn trong context.

3. Nếu context không chứa đủ thông tin để trả lời:
   "Tôi không thể xác minh thông tin này từ nguồn hiện có"

4. Trả lời bằng tiếng Việt.

5. Trình bày rõ ràng:
   - Tiêu đề nếu cần
   - Bullet point khi có nhiều ý

Không sử dụng kiến thức bên ngoài context.
"""


# =============================================================================
# DOCUMENT REORDERING
# =============================================================================


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp lại chunks để giảm Lost In The Middle.

    LLM thường chú ý tốt:
        - phần đầu prompt
        - phần cuối prompt

    Với input:

        [1,2,3,4,5]

    Trong đó:
        1 = score cao nhất
        5 = score thấp nhất


    Output:

        [1,3,5,4,2]


    Logic:

        front:
            chunks[::2]

        back:
            chunks[1::2]

        đảo back để chunk quan trọng thứ 2 nằm cuối.
    """

    if len(chunks) <= 2:
        return chunks


    # Các chunk thứ 1,3,5...
    front = chunks[::2]


    # Các chunk thứ 2,4...
    back = chunks[1::2]


    # Đưa chunk tốt thứ 2 xuống cuối context
    return front + back[::-1]



# =============================================================================
# CONTEXT FORMATTER
# =============================================================================


def format_context(chunks: list[dict]) -> str:
    """
    Convert retrieved chunks thành context gửi cho LLM.

    Input:
        [
            {
                content: "...",
                score: 0.8,
                metadata:{
                    source:"article_01.md"
                }
            }
        ]

    Output:

        [Document 1 | Source: article_01.md]

        Nội dung...

    """

    context_parts = []


    for index, chunk in enumerate(chunks, 1):

        metadata = chunk.get("metadata", {})


        source = metadata.get(
            "source",
            f"Document_{index}"
        )


        doc_type = metadata.get(
            "type",
            "unknown"
        )


        section = metadata.get(
            "section",
            ""
        )


        content = chunk.get(
            "content",
            ""
        )


        context_parts.append(
            f"""
[Document {index}]
Source: {source}
Type: {doc_type}
Section: {section}

{content}
"""
        )


    return "\n\n-----------------\n\n".join(context_parts)
# =============================================================================
# GENERATION PIPELINE
# =============================================================================


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:

        User Query
             |
             v
        Task 9 Retrieval
             |
             v
        Document Reordering
             |
             v
        Context Formatting
             |
             v
        LLM Generation
             |
             v
        Answer + Sources


    Returns:

    {
        "answer": str,
        "sources": list,
        "retrieval_source": str
    }

    """


    # ============================================================
    # Step 1: Retrieve context từ Task 9
    # ============================================================

    chunks = retrieve(
        query,
        top_k=top_k
    )


    # Không có context
    if not chunks:

        return {
            "answer":
                "Tôi không thể xác minh thông tin này từ nguồn hiện có",

            "sources": [],

            "retrieval_source":
                "none"
        }



    # ============================================================
    # Step 2: Reorder chunks
    # chống Lost In The Middle
    # ============================================================

    reordered_chunks = reorder_for_llm(
        chunks
    )



    # ============================================================
    # Step 3: Format context
    # ============================================================

    context = format_context(
        reordered_chunks
    )



    # ============================================================
    # Step 4: Build prompt
    # ============================================================

    user_prompt = f"""
Context:

{context}


----------------------------


Question:

{query}


Hãy trả lời câu hỏi dựa trên Context ở trên.
Mỗi thông tin quan trọng phải có citation.
Nếu Context không đủ thông tin, hãy nói:
"Tôi không thể xác minh thông tin này từ nguồn hiện có"
"""



    # ============================================================
    # Step 5: Call LLM
    # ============================================================

    try:

        from openai import OpenAI


        api_key = (
            os.getenv("OPENROUTER_API_KEY")
            or
            os.getenv("OPENAI_API_KEY")
        )


        if not api_key:

            raise RuntimeError(
                "Thiếu OPENROUTER_API_KEY hoặc OPENAI_API_KEY"
            )


        client = OpenAI(
            api_key=api_key,

            # OpenRouter dùng chuẩn OpenAI SDK
            base_url=
            "https://openrouter.ai/api/v1"
        )



        response = client.chat.completions.create(

            model=LLM_MODEL,


            messages=[

                {
                    "role": "system",

                    "content":
                    SYSTEM_PROMPT
                },


                {
                    "role": "user",

                    "content":
                    user_prompt
                }

            ],


            temperature=TEMPERATURE,


            top_p=TOP_P,

        )


        answer = (
            response
            .choices[0]
            .message
            .content
        )



    except Exception as e:


        print(
            f"⚠ LLM error: {e}"
        )


        answer = (
            "Tôi không thể xác minh thông tin này từ nguồn hiện có"
        )



    # ============================================================
    # Step 6: Return result
    # ============================================================


    retrieval_source = "hybrid"


    if chunks:

        retrieval_source = chunks[0].get(
            "source",
            "hybrid"
        )



    return {

        "answer":
            answer,


        "sources":
            reordered_chunks,


        "retrieval_source":
            retrieval_source
    }
# =============================================================================
# TEST
# =============================================================================


if __name__ == "__main__":

    test_queries = [

        "Shopee hỗ trợ những phương thức thanh toán nào?",

        "Làm sao để yêu cầu đổi trả hoặc hoàn tiền?",

        "Cần chuẩn bị bằng chứng gì khi yêu cầu hoàn tiền?",

    ]


    for query in test_queries:

        print("\n" + "=" * 80)

        print(
            f"QUESTION: {query}"
        )

        print("=" * 80)



        result = generate_with_citation(
            query,
            top_k=TOP_K
        )


        print("\nANSWER:")
        print(
            result["answer"]
        )


        print("\n" + "-" * 80)


        print(
            "RETRIEVAL SOURCE:",
            result["retrieval_source"]
        )


        print(
            "NUMBER OF SOURCES:",
            len(result["sources"])
        )


        print("\nSOURCES:")


        for i, source in enumerate(
            result["sources"],
            1
        ):

            metadata = source.get(
                "metadata",
                {}
            )


            print(
                f"""
{i}.
File:
{metadata.get('source')}

Score:
{source.get('score')}

Preview:
{source.get('content','')[:150]}...
"""
            )