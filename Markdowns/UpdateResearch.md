# Kế hoạch triển khai nâng cấp Research Mode theo 4 nhóm tài liệu cố định

## 1. Context và mục tiêu

### 1.1. Trạng thái hiện tại

Hệ thống Meibook hiện có 3 luồng hỏi đáp chính:

1. **MKAC / HCNS**
   - Dùng collection Qdrant riêng `mkac_knowledge`.
   - Retrieval cố định theo `session_id="mkac"`.
   - Có logic riêng trong `RAGPipeline._prepare_query_context()`.

2. **MES**
   - Tách khỏi RAG tài liệu.
   - Đi qua `MesQueryService`, `MesDatabase`, `MesSqlAgent`.

3. **Research**
   - Đã từng là luồng upload / nghiên cứu tài liệu.
   - Hiện đã bật lại ở frontend qua `ACTIVE_MODE_KEYS = ["mkac", "mes", "research"]` trong `frontend/src/main.jsx`.
   - Có demo cũ gồm 2 tài liệu tiếng Việt:
     - Script index: `scripts/index_research_demo_documents.py`
     - Collection: `docmind_documents`
     - Session cố định: `00000000-0000-4000-8000-000000000001`
     - Endpoint status: `GET /research/demo`
   - Research hiện query vào `self.vector_store` trong `src/rag/rag_pipeline.py`, tức collection mặc định `docmind_documents`, và dùng `session_id` từ request.

Gần đây đã OCR/index thêm bộ tài liệu Nhật:

- Source gốc: `documents/Research/DocJP/`
- Markdown OCR: `documents/Research/DocJP_md/`
- Manifest: `config/docjp_manifest.json`
- Script index: `scripts/index_docjp_documents.py`
- Collection mới: `docjp_knowledge`
- Session cố định trong collection: `docjp`
- Metadata quan trọng trong mỗi chunk:
  - `metadata.knowledge_base = "DocJP"`
  - `metadata.category = information_systems | legal_compliance | accounting | general_affairs`
  - `metadata.title`
  - `metadata.text_source`
  - checksum fields

### 1.2. Yêu cầu nghiệp vụ giai đoạn này

Giai đoạn này **không tập trung upload tài liệu mới**. Mục tiêu là nâng cấp Research mode để dùng bộ tài liệu có sẵn `DocJP`:

- Người dùng vào Research mode.
- Người dùng chọn **1 trong 4 nhóm tài liệu định nghĩa sẵn**:
  1. Công nghệ thông tin & Bảo mật (`information_systems`)
  2. Pháp chế & Quản lý rủi ro (`legal_compliance`)
  3. Kế toán (`accounting`)
  4. Hành chính tổng hợp (`general_affairs`)
- Sau khi chọn nhóm, mọi câu hỏi Research chỉ truy xuất tài liệu trong nhóm đó.
- Có thể có tùy chọn "Tất cả tài liệu" nếu cần search toàn bộ `DocJP`.
- Upload tài liệu riêng sẽ để giai đoạn sau, nhưng kiến trúc không được khóa chết khả năng đó.

### 1.3. Mục tiêu kỹ thuật

- Tận dụng collection `docjp_knowledge` hiện có, **không tách thành 4 collection vật lý**.
- Dùng Qdrant metadata filtering theo `metadata.category` để giới hạn phạm vi truy xuất.
- Tách rõ:
  - **Chat session**: phiên hội thoại UI.
  - **Knowledge scope**: collection/session/category tài liệu dùng để retrieval.
- Research không còn phụ thuộc demo session 2 file cũ.
- Research không còn hardcode model Grok; đi theo route `auto-model` / Qwen stack như hệ thống hiện tại.
- Frontend có UX rõ ràng, chuyên nghiệp: chọn nhóm tài liệu trước khi hỏi, hiển thị trạng thái index và danh sách nguồn.

---

## 2. Kiến trúc đề xuất

### 2.1. Nguyên tắc chính

**Dùng 1 collection Qdrant `docjp_knowledge` + metadata filter**, thay vì 4 collection nhỏ.

Lý do:

- 78 tài liệu là quy mô nhỏ đối với Qdrant.
- 1 collection giúp search toàn cục dễ hơn.
- Metadata filter theo `metadata.category` vẫn cho phép giới hạn phạm vi theo nhóm.
- Không cần viết router chọn collection phức tạp.
- Sau này thêm nhóm mới chỉ cần thêm metadata/config, không cần tạo collection mới.

### 2.2. Mô hình dữ liệu Research Topic

Tạo registry topic dạng config để backend và frontend dùng chung thông qua API.

Khuyến nghị tạo file mới:

```text
config/research_topics.json
```

Nội dung đề xuất:

```json
{
  "knowledge_base": "DocJP",
  "collection": "docjp_knowledge",
  "session_id": "docjp",
  "default_topic": "information_systems",
  "allow_all": true,
  "topics": [
    {
      "id": "information_systems",
      "category": "information_systems",
      "label_vi": "Công nghệ thông tin & Bảo mật",
      "label_ja": "情報システム・セキュリティ",
      "short_label_vi": "CNTT",
      "short_label_ja": "情報",
      "description_vi": "Hướng dẫn hệ thống IT, mạng, bảo mật, phần mềm, email, họp trực tuyến và các công cụ nội bộ.",
      "description_ja": "ITシステム、ネットワーク、セキュリティ、ソフトウェア、メール、Web会議、社内ツールに関する資料。",
      "icon": "shield",
      "accent": "blue",
      "quick_prompts_vi": [
        "Những phần mềm nào bị cấm sử dụng?",
        "Quy định sử dụng AI tạo sinh là gì?",
        "Cách xử lý khi nhận được email đáng ngờ?"
      ],
      "quick_prompts_ja": [
        "使用禁止ソフトには何がありますか？",
        "生成AIの利用ルールを教えてください。",
        "迷惑メールが届いた場合はどう対応しますか？"
      ]
    },
    {
      "id": "legal_compliance",
      "category": "legal_compliance",
      "label_vi": "Pháp chế & Quản lý rủi ro",
      "label_ja": "法務・リスク管理",
      "short_label_vi": "Pháp chế",
      "short_label_ja": "法務",
      "description_vi": "3rdWATCH, xác nhận an toàn, quản lý khủng hoảng, hợp đồng và con dấu.",
      "description_ja": "3rdWATCH、安否確認、危機管理、契約、捺印管理に関する資料。",
      "icon": "scale",
      "accent": "purple",
      "quick_prompts_vi": [
        "Cách đăng nhập 3rdWATCH như thế nào?",
        "Làm sao đăng ký email nhận thông báo an toàn?",
        "Quy trình đóng dấu hoặc quản lý con dấu ra sao?"
      ],
      "quick_prompts_ja": [
        "3rdWATCHへのログイン方法を教えてください。",
        "安否確認メールアドレスの登録方法は？",
        "捺印管理システムの使い方を教えてください。"
      ]
    },
    {
      "id": "accounting",
      "category": "accounting",
      "label_vi": "Kế toán",
      "label_ja": "経理",
      "short_label_vi": "Kế toán",
      "short_label_ja": "経理",
      "description_vi": "Tài liệu hướng dẫn và Q&A về hệ thống Rakuraku Seisan / thanh toán chi phí.",
      "description_ja": "楽楽精算の操作マニュアル、スマートフォン版、Q&A資料。",
      "icon": "calculator",
      "accent": "green",
      "quick_prompts_vi": [
        "Cách sử dụng Rakuraku Seisan trên điện thoại?",
        "Quy trình nộp thanh toán chi phí như thế nào?",
        "Có những câu hỏi thường gặp nào về Rakuraku Seisan?"
      ],
      "quick_prompts_ja": [
        "楽楽精算をスマートフォンで使う方法は？",
        "経費精算の申請手順を教えてください。",
        "楽楽精算のよくある質問をまとめてください。"
      ]
    },
    {
      "id": "general_affairs",
      "category": "general_affairs",
      "label_vi": "Hành chính tổng hợp",
      "label_ja": "総務",
      "short_label_vi": "Tổng vụ",
      "short_label_ja": "総務",
      "description_vi": "Tai nạn lao động, biểu mẫu tổng vụ, cấp phát vật tư, trang phục, quy định phê duyệt và cơ sở vật chất.",
      "description_ja": "労働災害、総務帳票、備品・作業服貸与、決裁権限、施設関連資料。",
      "icon": "building",
      "accent": "orange",
      "quick_prompts_vi": [
        "Khi xảy ra tai nạn lao động cần báo cáo thế nào?",
        "Cách xin cấp phát đồng phục hoặc giày bảo hộ?",
        "Bảng thẩm quyền phê duyệt quy định những gì?"
      ],
      "quick_prompts_ja": [
        "労働災害が発生した場合の報告手順は？",
        "作業服や作業靴の貸与依頼方法は？",
        "決裁権限基準表の内容を説明してください。"
      ]
    }
  ]
}
```

Có thể thêm topic đặc biệt:

```json
{
  "id": "all",
  "category": null,
  "label_vi": "Tất cả tài liệu",
  "label_ja": "すべての資料"
}
```

Nhưng nên coi `all` là lựa chọn phụ, không phải mặc định, vì câu hỏi chung trên 78 tài liệu dễ trả lời rộng và kém tập trung.

---

## 3. Backend plan

### 3.1. Sửa metadata hiện tại trước khi dùng filter

File liên quan:

```text
config/docjp_manifest.json
scripts/index_docjp_documents.py
```

Cần kiểm tra lại `config/docjp_manifest.json` vì manifest hiện được sinh từ logic nhận diện category bằng substring. Một số file có prefix `【法務】` nhưng trong tên có chữ `情報`, dẫn tới bị gán nhầm `information_systems`, ví dụ:

- `【法務】2-5.初期設定した情報の変更：ログインIDの変更.pdf`
- `【法務】2-6.初期設定した情報の変更：パスワードの変更.pdf`
- `【法務】2-8.ユーザ情報の確認.pdf`
- `【法務】4-1.危機管理情報を画面で確認する.pdf`
- `【法務】4-3.地震情報メールの受信設定.pdf`
- `【法務】4-4.鉄道運行情報メールの受信設定.pdf`
- `【法務】4-5.避難情報メールの受信設定.pdf`
- `【法務】4-6.情報メールの停止・配信設定の削除.pdf`

Yêu cầu sửa:

- Category phải lấy từ **prefix đầu tên file** dạng `^【(.+?)】`, không được tìm substring toàn bộ filename.
- Mapping chuẩn:
  - `【情報】` -> `information_systems`
  - `【法務】` -> `legal_compliance`
  - `【経理】` -> `accounting`
  - `【総務】` -> `general_affairs`
- Sau khi sửa manifest, phải chạy lại index cho DocJP để cập nhật metadata trong Qdrant:

```bash
MAX_DOCUMENT_PAGES=200 python scripts/index_docjp_documents.py --reindex
```

Nếu muốn tránh reindex toàn bộ, có thể reindex riêng những file category sai. Tuy nhiên để tránh checksum/metadata lệch, khuyến nghị reindex toàn bộ `DocJP` một lần.

### 3.2. Config constants

File:

```text
src/api/config.py
```

Thêm các env/config:

```python
DOCJP_COLLECTION_NAME = os.getenv("DOCJP_COLLECTION_NAME", "docjp_knowledge")
DOCJP_SESSION_ID = os.getenv("DOCJP_SESSION_ID", "docjp")
RESEARCH_TOPICS_PATH = Path(os.getenv("RESEARCH_TOPICS_PATH", "config/research_topics.json"))
RESEARCH_SCORE_THRESHOLD = float(os.getenv("RESEARCH_SCORE_THRESHOLD", "0.35"))
```

Ghi chú:

- `RESEARCH_SCORE_THRESHOLD` không nên dùng lại `MKAC_SCORE_THRESHOLD` vì tài liệu DocJP khác domain và ngôn ngữ.
- Có thể tune sau bằng regression prompt.

### 3.3. Schemas API

File:

```text
src/api/schemas.py
```

Thêm field vào `QueryRequest`:

```python
research_topic: Optional[str] = None
```

Ý nghĩa:

- `None` / missing: legacy behavior hoặc yêu cầu frontend không cho hỏi trước khi chọn topic.
- `information_systems`, `legal_compliance`, `accounting`, `general_affairs`: query theo topic.
- `all`: query toàn bộ collection `docjp_knowledge` không filter category.
- Sau này `custom`: query tài liệu upload riêng trong `docmind_documents` theo `session_id` như flow cũ.

Thêm response model:

```python
class ResearchTopic(BaseModel):
    id: str
    category: Optional[str] = None
    label_vi: str
    label_ja: str
    short_label_vi: str = ""
    short_label_ja: str = ""
    description_vi: str = ""
    description_ja: str = ""
    icon: str = "file_text"
    accent: str = "neutral"
    ready: bool = False
    num_files: int = 0
    num_chunks: int = 0
    files: List[str] = Field(default_factory=list)
    quick_prompts_vi: List[str] = Field(default_factory=list)
    quick_prompts_ja: List[str] = Field(default_factory=list)

class ResearchTopicsResponse(BaseModel):
    ready: bool
    collection: str
    session_id: str
    default_topic: Optional[str] = None
    allow_all: bool = True
    topics: List[ResearchTopic]
```

Có thể giữ `ResearchDemoResponse` để tương thích trong giai đoạn chuyển đổi, nhưng frontend mới nên dùng `/research/topics`.

### 3.4. Topic registry loader

Có 2 lựa chọn:

1. Đọc `config/research_topics.json` trực tiếp trong `src/api/main.py`.
2. Tạo module nhỏ `src/api/research_topics.py`.

Khuyến nghị tạo module mới:

```text
src/api/research_topics.py
```

Chức năng:

```python
def load_research_topic_config(path: Path) -> dict:
    ...

def research_topic_by_id(topic_id: str) -> dict | None:
    ...

def validate_research_topic(topic_id: str | None) -> str | None:
    ...
```

Yêu cầu:

- Nếu config thiếu hoặc JSON lỗi, endpoint `/research/topics` trả `ready=false` nhưng app không crash.
- Chỉ cho phép topic id trong registry hoặc `all`.
- Không nhận arbitrary category từ frontend để tránh query ngoài phạm vi định nghĩa.

### 3.5. VectorStore: hỗ trợ metadata filter và scroll phân trang

File:

```text
src/rag/vector_store.py
```

#### 3.5.1. Sửa `search()`

Hiện tại:

```python
def search(
    self,
    session_id: str,
    query_embedding: List[float],
    top_k: int = 5,
    score_threshold: float = 0.3,
) -> List[SearchResult]:
```

Đề xuất:

```python
def search(
    self,
    session_id: str,
    query_embedding: List[float],
    top_k: int = 5,
    score_threshold: float = 0.3,
    *,
    metadata_filters: Optional[Dict[str, Any]] = None,
) -> List[SearchResult]:
```

Filter build:

```python
must = [
    models.FieldCondition(
        key="session_id",
        match=models.MatchValue(value=session_id),
    )
]
for key, value in (metadata_filters or {}).items():
    must.append(
        models.FieldCondition(
            key=f"metadata.{key}",
            match=models.MatchValue(value=value),
        )
    )
```

Research call sẽ truyền:

```python
metadata_filters={"category": "information_systems"}
```

Cần verify Qdrant version hiện tại hỗ trợ nested key `metadata.category`. Nếu không hoạt động, fallback là reindex để đưa thêm `category` lên top-level payload trong `add_chunks()`. Nhưng ưu tiên thử nested key trước vì payload hiện đã có metadata.

#### 3.5.2. Sửa `get_session_info()` để scroll hết dữ liệu

Hiện `get_session_info()` dùng `scroll(limit=1000)` một lần. Với DocJP, file `●決裁権限基準表_結合_2025最新.xlsx` rất lớn, tổng chunks có thể vượt 1000, khiến count/status sai.

Cần refactor thành scroll phân trang:

```python
def iter_payloads(self, scroll_filter: models.Filter):
    offset = None
    while True:
        points, offset = self.client.scroll(..., offset=offset, limit=1000)
        yield from points
        if offset is None:
            break
```

Sau đó `get_session_info()` dùng helper này.

#### 3.5.3. Thêm `get_session_info(..., metadata_filters=None)`

Để endpoint `/research/topics` trả đúng số file/chunk theo category:

```python
def get_session_info(
    self,
    session_id: str,
    *,
    metadata_filters: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
```

Output vẫn giữ:

```python
{
  "num_chunks": int,
  "num_files": int,
  "files": List[str],
}
```

### 3.6. API startup: thêm `docjp_vector_store`

File:

```text
src/api/main.py
```

Hiện có:

```python
vector_store = VectorStore(host=QDRANT_HOST, port=QDRANT_PORT)
mkac_vector_store = VectorStore(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    collection_name=os.getenv("MKAC_COLLECTION_NAME", "mkac_knowledge"),
)
```

Thêm global:

```python
docjp_vector_store = None
```

Trong `lifespan()`:

```python
docjp_vector_store = VectorStore(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    collection_name=DOCJP_COLLECTION_NAME,
)
```

Truyền vào `RAGPipeline`:

```python
rag_pipeline = RAGPipeline(
    embedder=embedder,
    vector_store=vector_store,
    mkac_vector_store=mkac_vector_store,
    docjp_vector_store=docjp_vector_store,
    web_searcher=web_searcher,
    mes_database=mes_database,
)
```

### 3.7. Endpoint `/research/topics`

File:

```text
src/api/main.py
```

Thêm:

```python
@app.get("/research/topics", response_model=ResearchTopicsResponse)
async def research_topics_status():
    ...
```

Luồng:

1. Load `config/research_topics.json`.
2. Với từng topic:
   - gọi `docjp_vector_store.get_session_info(DOCJP_SESSION_ID, metadata_filters={"category": topic["category"]})`
   - lấy `num_files`, `num_chunks`, `files`
   - `ready = num_chunks > 0`
3. Nếu `allow_all=true`, có thể trả thêm topic `all`:
   - `category=None`
   - stats toàn bộ session `docjp`
4. Response:

```json
{
  "ready": true,
  "collection": "docjp_knowledge",
  "session_id": "docjp",
  "default_topic": "information_systems",
  "allow_all": true,
  "topics": [...]
}
```

Giữ `GET /research/demo` trong giai đoạn chuyển tiếp nhưng không dùng làm luồng chính.

### 3.8. Query routing: truyền `research_topic`

File:

```text
src/api/main.py
```

Sửa `route_query()` và `route_query_stream()`:

```python
return await rag_pipeline.query(
    session_id=req.session_id,
    question=routed_question,
    model=req.model,
    mode=req.mode,
    current_user=current_user_context,
    conversation_context=req.conversation_context,
    research_topic=req.research_topic,
)
```

Streaming tương tự.

### 3.9. RAGPipeline: thêm docjp_vector_store và Research scope resolver

File:

```text
src/rag/rag_pipeline.py
```

#### 3.9.1. Constructor

Hiện tại:

```python
def __init__(..., vector_store, mkac_vector_store=None, ...):
```

Thêm:

```python
docjp_vector_store: VectorStore | None = None,
```

và:

```python
self.docjp_vector_store = docjp_vector_store
```

#### 3.9.2. Query signatures

Thêm vào cả `query()` và `query_stream()`:

```python
research_topic: str | None = None,
```

#### 3.9.3. `_prepare_query_context()`

Hiện tại:

- `mode == "mkac"`: dùng `self.mkac_vector_store`, `session_id="mkac"`
- còn lại: dùng `self._retrieve(session_id, question, ...)`, tức research legacy dùng `self.vector_store`

Đề xuất tách nhánh rõ:

```python
if mode == "research":
    return self._prepare_research_query_context(
        session_id=session_id,
        question=question,
        research_topic=research_topic,
    )
```

Thêm method:

```python
def _prepare_research_query_context(
    self,
    *,
    session_id: str,
    question: str,
    research_topic: str | None,
) -> Tuple[List[SearchResult], List[Path], str]:
    ...
```

Luồng logic:

1. Nếu `research_topic` thuộc predefined topics hoặc `all`:
   - dùng `self.docjp_vector_store`
   - retrieval session id = `DOCJP_SESSION_ID` (`docjp`)
   - nếu topic != `all`, filter `metadata.category = topic.category`
   - answer_scope = `research`
2. Nếu `research_topic` missing:
   - Có 2 lựa chọn:
     - Strict: trả no result / raise error yêu cầu chọn topic.
     - Compatible: fallback legacy `self.vector_store.search(session_id=req.session_id)`.
   - Khuyến nghị giai đoạn đầu: frontend không cho hỏi khi chưa chọn topic; backend vẫn fallback legacy để không phá API cũ.
3. Nếu `research_topic == "custom"` trong tương lai:
   - dùng `self.vector_store`
   - retrieval session id = request `session_id`
   - không category filter

Pseudo:

```python
if research_topic and research_topic != "custom":
    topic = self._research_topic_config(research_topic)
    store = self.docjp_vector_store
    lookup_session_id = DOCJP_SESSION_ID
    metadata_filters = {}
    if topic.category:
        metadata_filters["category"] = topic.category
else:
    store = self.vector_store
    lookup_session_id = session_id
    metadata_filters = None

query_embedding = self.embedder.embed_query(question)
results = store.search(
    session_id=lookup_session_id,
    query_embedding=query_embedding,
    top_k=env_int("RESEARCH_TOP_K", 6, minimum=3, maximum=12),
    score_threshold=float(os.getenv("RESEARCH_SCORE_THRESHOLD", "0.35")),
    metadata_filters=metadata_filters,
)
```

#### 3.9.4. Model routing

Current issue: earlier analysis showed research was hardcoded to Grok in `_resolve_model()` in an older state. Current file should be verified; if any branch still routes `mode == "research"` to `grok-model`, remove it.

Target behavior:

- Text-only Research: `MODEL_ROUTES[model]`, default UI `auto` -> `auto-model`.
- Image-needed Research: only then route vision-capable model if needed.
- Since DocJP current index has markdown snippets and not page images, normal Research should not need vision.

#### 3.9.5. Prompt

File:

```text
src/rag/prompts.py
```

Update `RESEARCH_SYSTEM_PROMPT` to be stricter and suitable for internal Japanese docs:

- Answer in user language (`vi` or `ja`).
- Use only retrieved docs.
- Cite file and page.
- Separate:
  - Summary
  - Key findings / steps
  - Evidence
  - Unknown / missing info
- If evidence insufficient, say so.
- Do not invent policy beyond docs.
- Preserve Japanese product/system names, codes, email domains, URLs.

Also consider adding topic label to prompt via `build_rag_prompt()` metadata if available:

```text
Phạm vi tài liệu: Công nghệ thông tin & Bảo mật
```

This can come from `SearchResult.chunk.metadata.category` or explicit `research_topic`.

### 3.10. Source preview handling

Current `/sources/preview`:

- `mode == "mkac"` -> `mkac_vector_store`
- else research -> `vector_store` + request `session_id`

With DocJP predefined topics, sources live in `docjp_vector_store`, session `docjp`, not `vector_store`, so preview endpoint would be wrong if frontend tries to fetch page images.

Current DocJP index script uses:

```python
parser.process_file(..., image_output_dir=None, text_source=text_source)
```

So DocJP chunks likely do **not** have `metadata.image_path`; `format_sources()` returns `has_page_preview=false`. Therefore phase 1 can rely on snippet preview only.

Still, update plan should make preview robust:

1. Add source metadata in `format_sources()`:

```python
"knowledge_base": r.chunk.metadata.get("knowledge_base"),
"collection": "docjp_knowledge" if ... else ...,
"session_id": "docjp" if ... else ...,
```

2. Frontend:
   - If `source.has_page_preview === false`, open snippet dialog directly, do not call `/sources/preview`.

3. Backend optional future:
   - Extend `/sources/preview` with optional `knowledge_base=DocJP` or `collection=docjp_knowledge` to route to `docjp_vector_store`.
   - Add `DOCJP_PAGE_IMAGE_DIR` and reindex PDFs with page image output if page preview is required.

For this phase, **do not require DocJP page image preview**. Snippet + file/page citation is enough.

---

## 4. Frontend plan

File chính:

```text
frontend/src/main.jsx
frontend/src/styles.css
```

### 4.1. State mới

Thêm state:

```js
const [researchTopics, setResearchTopics] = useState({
  ready: false,
  collection: "",
  session_id: "",
  default_topic: "",
  topics: [],
});
const [researchTopicId, setResearchTopicId] = useState(() =>
  localStorage.getItem("meibook-research-topic") || ""
);
```

Nên scope theo language nếu title/label phụ thuộc language:

```js
const RESEARCH_TOPIC_STORAGE_KEY = "meibook-research-topic";
```

### 4.2. Bootstrap API

Trong `bootstrap()` hiện gọi:

```js
api("/research/demo")
```

Thêm hoặc thay bằng:

```js
api("/research/topics")
```

Giai đoạn chuyển tiếp:

- Có thể vẫn gọi `/research/demo` nếu muốn giữ nút demo cũ hidden/dev-only.
- UI chính dùng `researchTopics`.

### 4.3. Research empty state: Topic cards

Hiện empty state Research nếu `files.length === 0` hiển thị:

- Chọn tài liệu để bắt đầu
- Thử nghiệm với tài liệu mẫu

Giai đoạn này cần thay bằng:

- Tiêu đề: "Chọn nhóm tài liệu để bắt đầu nghiên cứu"
- Grid 4 topic cards:
  - Icon
  - Label
  - Description
  - `num_files` / `num_chunks`
  - Ready badge
- Optional card / button: "Tất cả tài liệu"
- Upload button tạm thời ẩn hoặc đưa xuống dưới dạng disabled:
  - "Upload tài liệu riêng — sẽ hỗ trợ ở giai đoạn sau"

Pseudo:

```jsx
{mode === "research" && !researchTopicId && (
  <ResearchTopicGrid
    topics={researchTopics.topics}
    language={language}
    onSelect={selectResearchTopic}
  />
)}
```

### 4.4. Chọn topic

Function:

```js
function selectResearchTopic(topicId) {
  setResearchTopicId(topicId);
  localStorage.setItem(RESEARCH_TOPIC_STORAGE_KEY, topicId);
  setModeMessages("research", [], language);
  setModeSources("research", [], language);
  setQuestion("");
  setError("");
}
```

Không nên đổi `sessionId` sang `docjp`, vì `sessionId` ở frontend đang dùng cho chat/session storage. Retrieval scope sẽ do backend quyết định bằng `research_topic`.

### 4.5. Điều kiện `canAsk`

Hiện:

```js
const researchReady = true;
```

Sửa thành:

```js
const researchReady =
  mode !== "research" || Boolean(researchTopicId && researchTopics.ready);
```

Nếu chưa chọn topic, input disabled / placeholder:

- VI: "Chọn nhóm tài liệu trước khi đặt câu hỏi..."
- JA: "質問する前に資料カテゴリを選択してください..."

### 4.6. Request payload

Trong `sendMessage()`, payload hiện có `mode`, `session_id`, `question`, `conversation_context`, ...

Thêm:

```js
research_topic: mode === "research" ? researchTopicId : null,
```

Streaming và non-streaming nếu có đều phải gửi field này.

### 4.7. Header / sidebar trạng thái topic

Trong Research mode sau khi chọn topic:

- Hiển thị badge ở top chat:

```text
資料調査 · 情報システム・セキュリティ · 32 tài liệu
```

- Có nút "Đổi nhóm tài liệu" / "カテゴリを変更".

Sidebar `document-sidebar` hiện chủ yếu phục vụ upload/list files. Giai đoạn này nên đổi nội dung khi `mode === "research"`:

- Nếu topic đã chọn:
  - Hiển thị topic description.
  - Danh sách file thuộc topic.
  - Số chunks.
- Nếu chưa chọn:
  - Hiển thị danh sách topic và hướng dẫn chọn.

Không xóa code upload; chỉ ẩn/disable phần upload để dễ bật lại sau.

### 4.8. Quick prompts theo topic

Hiện quick prompts Research nằm trong constant `QUICK_PROMPTS` chung:

```js
research: [
  "Lập báo cáo nghiên cứu tổng hợp từ các tài liệu",
  ...
]
```

Nên ưu tiên prompts từ `/research/topics`:

```js
function quickPromptsFor(workspaceMode, language) {
  if (workspaceMode === "research") {
    const topic = selectedResearchTopic();
    return language === "ja" ? topic.quick_prompts_ja : topic.quick_prompts_vi;
  }
  ...
}
```

Nếu chưa chọn topic, không hiển thị quick prompts.

### 4.9. UX copy đề xuất

VI:

- Empty title: `Nghiên cứu tài liệu nội bộ`
- Empty subtitle: `Chọn một nhóm tài liệu đã được index để bắt đầu. Kết quả sẽ chỉ dựa trên nhóm đã chọn.`
- Topic selected helper: `Đang nghiên cứu trong phạm vi: {topic}`
- Change topic: `Đổi nhóm tài liệu`
- Disabled input: `Chọn nhóm tài liệu trước khi đặt câu hỏi...`

JA:

- Empty title: `社内資料調査`
- Empty subtitle: `インデックス済み資料カテゴリを選択してください。回答は選択したカテゴリ内の資料に基づきます。`
- Topic selected helper: `調査範囲: {topic}`
- Change topic: `カテゴリを変更`
- Disabled input: `質問する前に資料カテゴリを選択してください...`

### 4.10. CSS

File:

```text
frontend/src/styles.css
```

Thêm classes:

```css
.research-topic-grid
.research-topic-card
.research-topic-card.selected
.research-topic-card.disabled
.research-topic-icon
.research-topic-meta
.research-topic-badge
.research-scope-bar
.research-topic-files
```

Yêu cầu UI:

- Desktop: grid 2x2 hoặc 4 columns tùy width.
- Mobile: single column.
- Card selected rõ màu.
- Có ready/not-ready badge.
- Dark mode tương thích.
- Không phá layout existing chat.

---

## 5. RAG behavior chi tiết

### 5.1. Query theo topic

Ví dụ user chọn `information_systems` và hỏi:

```text
使用禁止ソフトには何がありますか？
```

Flow:

1. Frontend gửi:

```json
{
  "mode": "research",
  "research_topic": "information_systems",
  "session_id": "<chat-session-id>",
  "question": "使用禁止ソフトには何がありますか？",
  "model": "auto",
  "ui_language": "ja"
}
```

2. Backend `route_query_stream()` gọi:

```python
rag_pipeline.query_stream(..., mode="research", research_topic="information_systems")
```

3. RAGPipeline chọn:

```text
store = docjp_vector_store
lookup_session_id = docjp
metadata.category = information_systems
```

4. Qdrant search:

```text
collection = docjp_knowledge
filter = session_id == docjp AND metadata.category == information_systems
```

5. Prompt chỉ nhận chunks thuộc IT/security docs.

6. Answer trả `answer_scope="research"`, sources gồm file/page/score/category/title.

### 5.2. Query all

Nếu user chọn `all`:

```text
collection = docjp_knowledge
filter = session_id == docjp
không filter category
```

Nên giới hạn top_k cao hơn một chút, ví dụ:

```text
RESEARCH_TOP_K=6 hoặc 8
```

### 5.3. Query custom future

Sau này khi upload tài liệu riêng:

```text
research_topic = custom
collection = docmind_documents
session_id = request.session_id
filter = session_id == request.session_id
```

Nhờ vậy upload flow cũ vẫn dùng lại được.

---

## 6. Cache và session

### 6.1. Query cache

Hiện `src/api/helpers.py::query_cache_key()` chỉ cache `mkac` và `mes`:

```python
if config.QUERY_RESPONSE_CACHE_SIZE <= 0 or req.mode not in {"mkac", "mes"}:
    return None
```

Vì Research chưa cache, không có nguy cơ cache nhầm topic.

Nếu sau này bật cache cho Research, key bắt buộc phải thêm:

- `research_topic`
- `DOCJP collection version` hoặc manifest checksum
- language/model/question

Ví dụ:

```python
return "|".join((
    req.mode,
    req.ui_language,
    req.model,
    req.research_topic or "",
    docjp_manifest_checksum,
    normalize_query_cache_text(req.question),
))
```

### 6.2. Session frontend

Không dùng `docjp` làm `session_id` frontend.

Lý do:

- `session_id` frontend hiện quản lý hội thoại / localStorage theo mode/language.
- `docjp` là namespace index trong Qdrant.
- Trộn 2 khái niệm sẽ gây lỗi khi sau này có upload custom.

Quy tắc:

- Frontend vẫn tạo session như hiện tại.
- `research_topic` quyết định knowledge scope.
- Backend tự map topic -> collection/session/category.

---

## 7. Migration / compatibility

### 7.1. Giữ demo cũ tạm thời

Không xóa ngay:

- `scripts/index_research_demo_documents.py`
- `/research/demo`
- `ResearchDemoResponse`
- `useResearchDemoSession()`

Nhưng UI chính không dùng nữa.

Có thể ẩn demo cũ bằng feature flag:

```env
RESEARCH_DEMO_ENABLED=false
```

Hoặc để nút demo trong dev-only nếu cần so sánh.

### 7.2. Upload hiện tại

Không phát triển thêm upload trong phase này.

Frontend:

- Ẩn hoặc disable upload CTA trong Research empty state.
- Không xóa code upload để phase sau tái dùng.

Backend:

- Giữ endpoint upload/session hiện tại.
- Không thay đổi behavior upload legacy ngoài việc không dùng từ UI chính.

---

## 8. Các bước triển khai cụ thể

### Phase 0 — Chuẩn hóa metadata DocJP

1. Sửa `config/docjp_manifest.json` để các file được gán category đúng theo prefix.
2. Nếu cần, viết script nhỏ hoặc sửa generator để parse prefix bằng regex `^【(.+?)】`.
3. Chạy reindex:

```bash
MAX_DOCUMENT_PAGES=200 python scripts/index_docjp_documents.py --reindex
```

4. Kiểm tra report:

```bash
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('logs/docjp_index_report.json').read_text(encoding='utf-8'))
print('indexed', len(report.get('indexed', [])))
print('failed', report.get('failed'))
PY
```

### Phase 1 — Backend data/config

1. Tạo `config/research_topics.json`.
2. Sửa `src/api/config.py` thêm `DOCJP_COLLECTION_NAME`, `DOCJP_SESSION_ID`, `RESEARCH_TOPICS_PATH`, `RESEARCH_SCORE_THRESHOLD`.
3. Tạo `src/api/research_topics.py` để load/validate registry.
4. Sửa `src/api/schemas.py` thêm:
   - `QueryRequest.research_topic`
   - `ResearchTopic`
   - `ResearchTopicsResponse`

### Phase 2 — VectorStore filters/stats

1. Sửa `VectorStore.search()` thêm `metadata_filters`.
2. Sửa `get_session_info()` scroll phân trang.
3. Thêm `metadata_filters` cho `get_session_info()`.
4. Test bằng Qdrant local:

```bash
curl -fsS http://localhost:6333/collections/docjp_knowledge | jq .
```

và bằng Python/pytest để đảm bảo filter `metadata.category` trả đúng files/chunks.

### Phase 3 — Backend Research API/RAG

1. Sửa `src/api/main.py`:
   - import config mới
   - global `docjp_vector_store`
   - init `docjp_vector_store` trong lifespan
   - truyền vào `RAGPipeline`
   - thêm endpoint `/research/topics`
   - truyền `req.research_topic` vào `rag_pipeline.query()` và `query_stream()`

2. Sửa `src/rag/rag_pipeline.py`:
   - constructor nhận `docjp_vector_store`
   - `query()` / `query_stream()` nhận `research_topic`
   - `_prepare_query_context()` tách nhánh research rõ ràng
   - thêm `_prepare_research_query_context()`
   - đảm bảo `_resolve_model()` không ép research sang `grok-model`

3. Sửa `src/rag/prompts.py` cập nhật `RESEARCH_SYSTEM_PROMPT`.

4. Nếu cần, sửa `format_sources()` để thêm `knowledge_base`, `category`, `source_session_id` cho frontend.

### Phase 4 — Frontend UX

1. Sửa `frontend/src/main.jsx`:
   - thêm state `researchTopics`, `researchTopicId`
   - bootstrap gọi `/research/topics`
   - thêm helper `selectedResearchTopic()`
   - thêm function `selectResearchTopic(topicId)` và `clearResearchTopic()`
   - sửa `canAsk` / placeholder research theo topic
   - thêm `research_topic` vào payload query
   - thay empty state Research bằng topic grid
   - sidebar hiển thị topic info/files thay vì upload là chính
   - quick prompts lấy từ topic config

2. Sửa `frontend/src/styles.css`:
   - topic cards
   - research scope bar
   - responsive mobile
   - dark mode

### Phase 5 — Docs/tests

1. Cập nhật `Markdowns/MesEXPLAIN.md` hoặc tài liệu vận hành tương ứng:
   - Research mode mới
   - collection `docjp_knowledge`
   - 4 topics
   - cách reindex

2. Cập nhật `Markdowns/DEPLOY.md`:
   - `DOCJP_COLLECTION_NAME`
   - `DOCJP_SESSION_ID`
   - `RESEARCH_TOPICS_PATH`
   - lệnh index DocJP

3. Thêm tests.

---

## 9. Test plan

### 9.1. Unit tests

#### `tests/test_research_topics.py`

Test:

- Load config hợp lệ.
- Validate 4 topic id.
- Reject topic id không tồn tại.
- `all` được chấp nhận nếu `allow_all=true`.
- Labels VI/JA tồn tại.

#### `tests/test_vector_store_filters.py`

Nếu có thể mock Qdrant client:

- `search(..., metadata_filters={"category": "accounting"})` build filter gồm:
  - `session_id == docjp`
  - `metadata.category == accounting`

Nếu test integration Qdrant không tiện, test helper build filter riêng.

#### `tests/test_query_routing.py`

Thêm schema test:

- `QueryRequest(mode="research", research_topic="accounting")` valid.
- Missing `research_topic` vẫn valid để backward compatible.

### 9.2. Backend integration check

Sau khi app chạy:

```bash
curl -fsS http://localhost:8001/research/topics | jq .
```

Kỳ vọng:

- `ready: true`
- Có 4 topics.
- Mỗi topic `num_files > 0`, `num_chunks > 0`.
- Category pháp chế không bị thiếu do manifest category sai.

Test query từng topic:

```bash
SESSION_ID=$(curl -fsS -X POST http://localhost:8001/sessions | jq -r .session_id)

curl -fsS -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"question\": \"使用禁止ソフトには何がありますか？\",
    \"model\": \"auto\",
    \"mode\": \"research\",
    \"ui_language\": \"ja\",
    \"research_topic\": \"information_systems\"
  }" | jq .
```

Kiểm tra:

- `answer_scope == "research"`
- sources đều có `category == "information_systems"`
- model không phải `grok-model` trừ khi explicit chọn grok.

Các câu test mẫu:

1. `information_systems`
   - JA: `使用禁止ソフトには何がありますか？`
   - VI: `Những phần mềm nào bị cấm sử dụng?`

2. `legal_compliance`
   - JA: `3rdWATCHへのログイン方法を教えてください。`
   - VI: `Cách đăng nhập 3rdWATCH như thế nào?`

3. `accounting`
   - JA: `楽楽精算をスマートフォンで使う方法は？`
   - VI: `Cách dùng Rakuraku Seisan trên điện thoại?`

4. `general_affairs`
   - JA: `労働災害が発生した場合の報告手順は？`
   - VI: `Khi xảy ra tai nạn lao động cần báo cáo thế nào?`

### 9.3. Frontend verification

1. Build frontend:

```bash
cd frontend
npm run build
cd ..
```

2. Mở UI:

```bash
http://localhost:8001
```

Checklist:

- Research tab hiển thị.
- Empty state Research hiển thị 4 topic cards.
- Chưa chọn topic thì input disabled hoặc có placeholder yêu cầu chọn topic.
- Chọn topic xong:
  - scope bar hiện đúng topic.
  - quick prompts đổi theo topic.
  - hỏi được câu hỏi.
  - sources hiển thị file/page/snippet.
- Đổi topic:
  - clear hoặc reset conversation rõ ràng.
  - không giữ sources cũ gây nhầm.
- UI JA/VN đổi label đúng.
- Mobile responsive.

### 9.4. Regression tests không được bỏ qua

Chạy tối thiểu:

```bash
pytest tests/test_query_routing.py
pytest tests/test_employee_directory.py
pytest tests/test_mes_database.py
pytest tests/test_mes_sql_agent.py
```

Nếu môi trường local thiếu dependency như `numpy/torch/fitz/fastapi`, ghi rõ test nào không chạy được và lý do.

---

## 10. Rủi ro và cách xử lý

### 10.1. Qdrant nested metadata filter không hoạt động

Rủi ro:

- `metadata.category` có thể không match do version client/Qdrant.

Cách xử lý:

1. Test trực tiếp bằng search/scroll.
2. Nếu fail, sửa `VectorStore.add_chunks()` thêm top-level payload:

```python
"category": chunk.metadata.get("category"),
"knowledge_base": chunk.metadata.get("knowledge_base"),
```

3. Reindex DocJP.
4. Filter bằng key `category` thay vì `metadata.category`.

### 10.2. Manifest category hiện bị sai

Rủi ro:

- Một số file `【法務】...情報...` đang bị gán `information_systems`.

Cách xử lý:

- Bắt buộc sửa manifest theo prefix trước khi triển khai topic filter.
- Reindex.

### 10.3. `get_session_info()` chỉ scroll 1000 chunks

Rủi ro:

- Topic stats sai, nhất là file Excel lớn.

Cách xử lý:

- Scroll phân trang.

### 10.4. Source preview sai collection

Rủi ro:

- Frontend gọi `/sources/preview` cho DocJP nhưng endpoint tìm trong `docmind_documents` thay vì `docjp_knowledge`.

Cách xử lý phase này:

- Nếu `has_page_preview=false`, frontend chỉ mở snippet, không fetch image preview.
- Phase sau nếu cần image preview: thêm routing preview theo `knowledge_base`/`collection` và index page images cho DocJP.

### 10.5. Lẫn chat session và knowledge session

Rủi ro:

- Dùng `session_id=docjp` ở frontend sẽ làm hỏng session history.

Cách xử lý:

- Frontend session giữ nguyên.
- Backend tự map `research_topic` -> `DOCJP_SESSION_ID`.

### 10.6. Research model route dùng Grok

Rủi ro:

- Nếu còn hardcode `mode == research -> grok-model`, latency/cost sai và lệch Qwen stack.

Cách xử lý:

- Kiểm tra `_resolve_model()`.
- Text Research dùng `auto-model`.
- Chỉ dùng vision model khi có image input thật.

### 10.7. Tiếng Nhật retrieval/answer

Rủi ro:

- Query tiếng Việt hỏi tài liệu Nhật có thể retrieval kém hơn query tiếng Nhật.

Cách xử lý phase này:

- BGE-M3 multilingual nên vẫn có thể chạy.
- Nếu chất lượng VI->JA thấp, thêm bước optional query rewrite/localize cho Research:
  - UI VI + topic DocJP: rewrite question sang Japanese for retrieval, nhưng answer vẫn tiếng Việt.
- Chưa implement ngay nếu chưa thấy lỗi thực tế.

---

## 11. Định nghĩa hoàn thành (Definition of Done)

Research mode được coi là hoàn thành khi:

1. `/research/topics` trả 4 topic với stats đúng.
2. Frontend Research cho chọn 1 trong 4 topic.
3. Query Research truyền `research_topic` xuống backend.
4. Backend query `docjp_knowledge` session `docjp` và filter đúng category.
5. Sources trả về đều thuộc category đã chọn.
6. Research dùng `auto-model` mặc định, không ép Grok.
7. Không phá MKAC/MES hiện tại.
8. Upload legacy chưa cần dùng nhưng code không bị xóa/khóa chết.
9. Frontend build thành công.
10. Có test hoặc manual verification cho cả 4 topic.

---

## 12. Checklist triển khai cho Codex review

- [x] `config/docjp_manifest.json` category đúng theo prefix `【...】`.
- [x] `config/research_topics.json` tồn tại và có 4 topics.
- [x] `src/api/config.py` có config DocJP/Research topics.
- [x] `src/api/schemas.py` có `research_topic` và response schemas mới.
- [x] `src/api/research_topics.py` hoặc logic loader tương đương có validation topic id.
- [x] `src/rag/vector_store.py::search()` hỗ trợ metadata filters.
- [x] `src/rag/vector_store.py::get_session_info()` scroll phân trang và filter theo metadata.
- [x] `src/api/main.py` tạo `docjp_vector_store` và endpoint `/research/topics`.
- [x] `src/api/main.py` truyền `research_topic` vào RAGPipeline cho query + stream.
- [x] `src/rag/rag_pipeline.py` có nhánh Research riêng dùng `docjp_vector_store`.
- [x] `_resolve_model()` không hardcode Research sang Grok.
- [x] `src/rag/prompts.py` Research prompt được cập nhật.
- [x] `frontend/src/main.jsx` có topic selector, selected topic state, payload `research_topic`.
- [x] `frontend/src/styles.css` có CSS responsive cho topic cards.
- [ ] Upload UI tạm ẩn/disable nhưng code upload legacy còn giữ.
- [ ] Source snippet hoạt động khi không có page preview.
- [ ] Tests/manual curl cho 4 topic pass.
- [x] Frontend build pass.

Trạng thái runtime đã kiểm tra:

- `/research/topics` trả `ready=true`, collection `docjp_knowledge`, session `docjp`.
- 4 topic đều có thống kê file/chunk từ Qdrant.
- `/research/demo` vẫn sẵn sàng với session
  `00000000-0000-4000-8000-000000000001`, 2 file và 39 chunk.
- Frontend hiện phục vụ asset mới `index-Col7hhdo.js` và `index-DJL9emDw.css`.
