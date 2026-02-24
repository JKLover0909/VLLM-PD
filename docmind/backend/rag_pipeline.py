"""
rag_pipeline.py
---------------
RAG orchestration: nhận câu hỏi → tìm context → tạo prompt → gọi LLM.
Prompt được tối ưu cho tiếng Việt + tiếng Anh song ngữ.
"""

import logging
from typing import AsyncGenerator, Optional

from embedder import Embedder
from vector_store import VectorStore, SearchResult
from vllm_client import VLLMClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Bạn là trợ lý phân tích tài liệu thông minh, hỗ trợ cả tiếng Việt và tiếng Anh.

Nguyên tắc:
1. Chỉ trả lời dựa trên thông tin trong các đoạn tài liệu được cung cấp.
2. Trả lời bằng ngôn ngữ của câu hỏi (nếu hỏi tiếng Việt → trả lời tiếng Việt, hỏi tiếng Anh → trả lời tiếng Anh).
3. Luôn trích dẫn nguồn ở cuối câu trả lời, ví dụ: [Nguồn: tên_file.pdf, trang 3].
4. Nếu thông tin không có trong tài liệu, hãy nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
5. Trình bày rõ ràng, có cấu trúc. Dùng gạch đầu dòng khi phù hợp.
6. Không bịa đặt thông tin ngoài tài liệu."""


def build_rag_prompt(question: str, search_results: list[SearchResult]) -> list[dict]:
    """
    Tạo messages list cho vLLM từ câu hỏi và context tìm được.

    Format:
        system: hướng dẫn hành vi
        user: [context] + câu hỏi
    """
    if not search_results:
        context_text = "(Không tìm thấy đoạn tài liệu liên quan.)"
    else:
        context_parts = []
        for i, result in enumerate(search_results, 1):
            c = result.chunk
            citation = f"[{c.source_file}, trang {c.page_number + 1}]"
            context_parts.append(
                f"--- Đoạn {i} {citation} ---\n{c.text.strip()}"
            )
        context_text = "\n\n".join(context_parts)

    user_message = (
        f"Dưới đây là các đoạn trích từ tài liệu:\n\n"
        f"{context_text}\n\n"
        f"---\n"
        f"Câu hỏi: {question}\n\n"
        f"Hãy trả lời câu hỏi dựa trên các đoạn tài liệu trên."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


class RAGPipeline:
    """
    Điều phối toàn bộ RAG pipeline:
    Question → Embed → Search → Build Prompt → LLM → Answer
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        vllm_client: VLLMClient,
        top_k: int = 5,
        score_threshold: float = 0.25,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.vllm_client = vllm_client
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def query(
        self,
        session_id: str,
        question: str,
    ) -> tuple[str, list[SearchResult]]:
        """
        Non-streaming: trả về (answer, search_results).
        """
        search_results = self._retrieve(session_id, question)
        messages = build_rag_prompt(question, search_results)

        answer = await self.vllm_client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return answer, search_results

    async def query_stream(
        self,
        session_id: str,
        question: str,
    ) -> tuple[AsyncGenerator[str, None], list[SearchResult]]:
        """
        Streaming: trả về (token_generator, search_results).
        search_results có ngay, tokens được stream dần.
        """
        search_results = self._retrieve(session_id, question)
        messages = build_rag_prompt(question, search_results)

        token_stream = self.vllm_client.chat_stream(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return token_stream, search_results

    def _retrieve(self, session_id: str, question: str) -> list[SearchResult]:
        """Embed câu hỏi và tìm top-k chunks liên quan."""
        query_embedding = self.embedder.embed_query(question)
        results = self.vector_store.search(
            session_id=session_id,
            query_embedding=query_embedding,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )
        logger.info(
            f"Retrieved {len(results)} chunks for query: '{question[:50]}...'"
        )
        return results

    def format_sources(self, results: list[SearchResult]) -> list[dict]:
        """Format search results thành list dict để trả về API."""
        return [
            {
                "file": r.chunk.source_file,
                "page": r.chunk.page_number + 1,
                "score": round(r.score, 4),
                "type": r.chunk.content_type,
                "preview": r.chunk.text[:200] + "..."
                if len(r.chunk.text) > 200
                else r.chunk.text,
            }
            for r in results
        ]
