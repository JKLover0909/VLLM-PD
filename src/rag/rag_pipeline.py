"""
src/rag/rag_pipeline.py
-----------------------
Điều phối toàn bộ RAG pipeline cho Máy 2:
Nhận câu hỏi -> Embed -> Tìm kiếm ngữ nghĩa trong Qdrant -> Tạo prompt đa phương thức -> Gọi LiteLLM.
"""

import logging
import base64
import os
from pathlib import Path
from typing import AsyncGenerator, Tuple, List, Dict, Any
from openai import AsyncOpenAI

from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý phân tích tài liệu thông minh, hỗ trợ cả tiếng Việt và tiếng Anh.

Nguyên tắc trả lời:
1. Chỉ trả lời dựa trên các đoạn thông tin từ tài liệu và hình ảnh (nếu có) được cung cấp.
2. Trả lời bằng ngôn ngữ của câu hỏi (nếu hỏi bằng tiếng Việt -> trả lời tiếng Việt, hỏi tiếng Anh -> trả lời tiếng Anh).
3. Luôn trích dẫn tên tệp và số trang ở cuối phần thông tin liên quan, ví dụ: [Nguồn: file_name.pdf, trang 3].
4. Nếu thông tin không xuất hiện trong tài liệu, hãy nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
5. Trình bày thông tin rõ ràng, có cấu trúc, sử dụng gạch đầu dòng khi cần thiết.
6. Tuyệt đối không tự suy diễn hoặc bịa đặt thông tin nằm ngoài tài liệu."""


def build_rag_prompt(question: str, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
    """
    Tạo danh sách messages cho OpenAI client từ câu hỏi và context tìm được.
    Hỗ trợ text và hình ảnh (Vision).
    """
    if not search_results:
        context_text = "(Không tìm thấy đoạn tài liệu liên quan.)"
    else:
        context_parts = []
        for i, result in enumerate(search_results, 1):
            c = result.chunk
            citation = f"[{c.source_file}, trang {c.page_number}]"
            context_parts.append(
                f"--- Đoạn {i} {citation} ---\n{c.text.strip()}"
            )
        context_text = "\n\n".join(context_parts)

    user_message = (
        f"Dưới đây là các đoạn trích từ tài liệu:\n\n"
        f"{context_text}\n\n"
        f"---\n"
        f"Câu hỏi: {question}\n\n"
        f"Hãy trả lời câu hỏi dựa trên các đoạn tài liệu và hình ảnh đính kèm (nếu có) ở trên."
    )

    image_content = _build_image_content(search_results)

    if image_content:
        user_content = [{"type": "text", "text": user_message}] + image_content
    else:
        user_content = user_message

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_image_content(search_results: List[SearchResult], max_images: int = 2) -> List[Dict[str, Any]]:
    """
    Quét qua các metadata của search results để tải ảnh và chuyển đổi sang dạng base64 gửi cho VLM.
    """
    image_items = []
    seen_paths = set()

    for result in search_results:
        metadata = result.chunk.metadata
        image_path = metadata.get("image_path") if metadata else None
        if not image_path or image_path in seen_paths:
            continue

        path = Path(image_path)
        if not path.exists():
            continue

        try:
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            suffix = path.suffix.lower().replace(".", "") or "png"
            mime = "jpeg" if suffix == "jpg" else suffix
            image_items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{encoded}"},
                }
            )
            seen_paths.add(image_path)
        except Exception as e:
            logger.warning(f"Error encoding image for Vision model: {e}")
            continue

        if len(image_items) >= max_images:
            break

    return image_items


class RAGPipeline:
    """
    Điều phối RAG: Nhận câu hỏi -> Embed -> Tìm kiếm ngữ nghĩa -> Gọi LiteLLM.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.25,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Kết nối tới LiteLLM Proxy
        proxy_url = os.getenv("LITELLM_URL", "http://localhost:4000/v1")
        self.openai_client = AsyncOpenAI(
            api_key="sk-local",
            base_url=proxy_url
        )

    async def query(self, session_id: str, question: str) -> Tuple[str, List[SearchResult]]:
        """
        Non-streaming RAG query.
        """
        search_results = self._retrieve(session_id, question)
        messages = build_rag_prompt(question, search_results)

        try:
            # Dùng vision-model để tự động gọi local Qwen3-VL hoặc fallback sang OpenAI GPT-4o
            response = await self.openai_client.chat.completions.create(
                model="vision-model",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content
            return answer, search_results
        except Exception as e:
            logger.error(f"Error in RAG generation: {e}")
            raise e

    async def query_stream(self, session_id: str, question: str) -> Tuple[AsyncGenerator[str, None], List[SearchResult]]:
        """
        Streaming RAG query.
        """
        search_results = self._retrieve(session_id, question)
        messages = build_rag_prompt(question, search_results)

        try:
            response = await self.openai_client.chat.completions.create(
                model="vision-model",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            async def token_generator():
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

            return token_generator(), search_results
        except Exception as e:
            logger.error(f"Error in streaming RAG generation: {e}")
            raise e

    def _retrieve(self, session_id: str, question: str) -> List[SearchResult]:
        """
        Tìm kiếm ngữ nghĩa từ Vector Store.
        """
        query_embedding = self.embedder.embed_query(question)
        results = self.vector_store.search(
            session_id=session_id,
            query_embedding=query_embedding,
            top_k=self.top_k,
            score_threshold=self.score_threshold
        )
        logger.info(f"Retrieved {len(results)} chunks for query: '{question[:50]}'")
        return results

    def format_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """
        Định dạng nguồn trích dẫn trả về API.
        """
        return [
            {
                "file": r.chunk.source_file,
                "page": r.chunk.page_number,
                "score": round(r.score, 4),
                "type": r.chunk.content_type,
                "preview": r.chunk.text[:200] + "..." if len(r.chunk.text) > 200 else r.chunk.text
            }
            for r in results
        ]
