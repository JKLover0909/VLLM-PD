"""
src/rag/rag_pipeline.py
-----------------------
Điều phối toàn bộ RAG pipeline cho Máy 2:
Nhận câu hỏi -> Embed -> Tìm kiếm ngữ nghĩa trong Qdrant -> Tạo prompt đa phương thức -> Gọi LiteLLM.
"""

import asyncio
import base64
import logging
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

RESEARCH_SYSTEM_PROMPT = """Bạn là chuyên gia nghiên cứu tài liệu, hỗ trợ cả tiếng Việt và tiếng Anh.

Nguyên tắc:
1. Chỉ sử dụng bằng chứng từ các đoạn tài liệu được cung cấp.
2. Tổng hợp theo cấu trúc: Tóm tắt điều hành, Phát hiện chính, Bằng chứng, Điểm chưa rõ và Câu hỏi nghiên cứu tiếp theo.
3. Phân biệt rõ dữ kiện, suy luận có căn cứ và thông tin còn thiếu.
4. Trích dẫn tên tệp và số trang cho từng phát hiện quan trọng.
5. Trả lời bằng ngôn ngữ của câu hỏi.
6. Không bịa đặt hoặc bổ sung kiến thức ngoài tài liệu."""

MODEL_ROUTES = {
    "auto": "auto-model",
    "local": "local-gemma",
    "mimo": "mimo-pro",
    "openai": "openai-model",
}

LOCAL_MODEL_ALIASES = {"local-gemma", "coding-model"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def build_rag_prompt(
    question: str,
    search_results: List[SearchResult],
    mode: str = "chat",
    image_paths: List[Path] | None = None,
) -> List[Dict[str, Any]]:
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

    instruction = (
        "Hãy lập báo cáo nghiên cứu dựa trên các đoạn tài liệu và hình ảnh đính kèm (nếu có)."
        if mode == "research"
        else "Hãy trả lời câu hỏi dựa trên các đoạn tài liệu và hình ảnh đính kèm (nếu có) ở trên."
    )
    user_message = (
        f"Dưới đây là các đoạn trích từ tài liệu:\n\n"
        f"{context_text}\n\n"
        f"---\n"
        f"Câu hỏi: {question}\n\n"
        f"{instruction}"
    )

    image_content = _build_image_content(search_results, image_paths=image_paths)

    if image_content:
        user_content = [{"type": "text", "text": user_message}] + image_content
    else:
        user_content = user_message

    return [
        {
            "role": "system",
            "content": RESEARCH_SYSTEM_PROMPT if mode == "research" else SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_content},
    ]


def _build_image_content(
    search_results: List[SearchResult],
    max_images: int = 2,
    image_paths: List[Path] | None = None,
) -> List[Dict[str, Any]]:
    """
    Quét qua các metadata của search results để tải ảnh và chuyển đổi sang dạng base64 gửi cho VLM.
    """
    image_items = []
    seen_paths = set()

    paths: List[Path] = []
    if image_paths:
        paths.extend(image_paths)

    for result in search_results:
        metadata = result.chunk.metadata
        image_path = metadata.get("image_path") if metadata else None
        if not image_path or image_path in seen_paths:
            continue
        paths.append(Path(image_path))

    for path in paths:
        image_path = str(path)
        if image_path in seen_paths:
            continue
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
            api_key=os.getenv("LITELLM_MASTER_KEY", "sk-local"),
            base_url=proxy_url
        )

    async def query(
        self,
        session_id: str,
        question: str,
        model: str = "auto",
        mode: str = "chat",
    ) -> Tuple[str, List[SearchResult], str]:
        """
        Non-streaming RAG query.
        """
        search_results = await asyncio.to_thread(
            self._retrieve,
            session_id,
            question,
            10 if mode == "research" else self.top_k,
        )
        image_paths = self._session_image_paths(session_id)
        messages = build_rag_prompt(
            question,
            search_results,
            mode=mode,
            image_paths=image_paths,
        )
        routed_model = self._resolve_model(model, has_images=bool(image_paths))

        try:
            response = await self.openai_client.chat.completions.create(
                model=routed_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1800 if mode == "research" else self.max_tokens,
                **self._provider_options(routed_model),
            )
            answer = response.choices[0].message.content or ""
            return answer, search_results, routed_model
        except Exception as e:
            logger.error(f"Error in RAG generation: {e}")
            raise e

    async def query_stream(
        self,
        session_id: str,
        question: str,
        model: str = "auto",
        mode: str = "chat",
    ) -> Tuple[AsyncGenerator[str, None], List[SearchResult], str]:
        """
        Streaming RAG query.
        """
        search_results = await asyncio.to_thread(
            self._retrieve,
            session_id,
            question,
            10 if mode == "research" else self.top_k,
        )
        image_paths = self._session_image_paths(session_id)
        messages = build_rag_prompt(
            question,
            search_results,
            mode=mode,
            image_paths=image_paths,
        )
        routed_model = self._resolve_model(model, has_images=bool(image_paths))

        try:
            response = await self.openai_client.chat.completions.create(
                model=routed_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1800 if mode == "research" else self.max_tokens,
                stream=True,
                **self._provider_options(routed_model),
            )

            async def token_generator():
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

            return token_generator(), search_results, routed_model
        except Exception as e:
            logger.error(f"Error in streaming RAG generation: {e}")
            raise e

    def _retrieve(
        self,
        session_id: str,
        question: str,
        top_k: int | None = None,
    ) -> List[SearchResult]:
        """
        Tìm kiếm ngữ nghĩa từ Vector Store.
        """
        query_embedding = self.embedder.embed_query(question)
        results = self.vector_store.search(
            session_id=session_id,
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
            score_threshold=self.score_threshold
        )
        logger.info(f"Retrieved {len(results)} chunks for query: '{question[:50]}'")
        return results

    def _resolve_model(self, model: str, has_images: bool = False) -> str:
        """Ánh xạ lựa chọn từ UI sang model logic của LiteLLM."""
        if has_images:
            return "openai-model"
        try:
            return MODEL_ROUTES[model]
        except KeyError as exc:
            raise ValueError(f"Unsupported model option: {model}") from exc

    def _session_image_paths(self, session_id: str) -> List[Path]:
        """Return uploaded image paths for a session, if any."""
        info = self.vector_store.get_session_info(session_id)
        if not info:
            return []

        upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
        session_dir = upload_dir / session_id
        paths = []
        for filename in info.get("files", []):
            path = session_dir / Path(filename).name
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.exists():
                paths.append(path.resolve())
        return paths

    @staticmethod
    def _provider_options(routed_model: str) -> Dict[str, Any]:
        """Provider-specific safeguards for LiteLLM upstream models."""
        if routed_model in LOCAL_MODEL_ALIASES:
            # Gemma4 on Ollama may spend the whole generation budget in
            # message.thinking, leaving message.content empty or truncated.
            return {"extra_body": {"think": False}}
        return {}

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
