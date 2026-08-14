# Meibook / VLLM-PD: Tài liệu chuẩn bị phỏng vấn AI Engineer

> **Mục tiêu:** hiểu hệ thống từ câu hỏi người dùng đến câu trả lời, giải thích được các trade-off về LLM, RAG, agent và LangGraph bằng chính kinh nghiệm triển khai trong Meibook.
>
> **Cách dùng:** đọc phần 1 để có “bản đồ”; đọc phần 2–5 để nắm kỹ thuật; cuối cùng tự trả lời phần 6 mà không nhìn đáp án.
>
> **Phạm vi:** các nhận định “Meibook đang dùng” dựa trên source code trong repository tại ngày **2026-08-13**. Các liên kết trong phần “Nguồn chính thức” là tài liệu để học khái niệm, không có nghĩa Meibook đã dùng mọi tính năng được mô tả.

---

## 1. Bản đồ hệ thống trong 90 giây

```text
React UI
  │ POST /query hoặc POST /query/stream
  ▼
FastAPI API
  │ xác thực, chuẩn hóa request, chọn mode
  ├── HR/MKAC ── structured intent → SQLite
  │              nếu không khớp → BGE-M3 → Qdrant mkac_knowledge → prompt
  ├── MES ────── deterministic SQL/API
  │              → LLM SQL agent (fallback có guardrails)
  ├── Research ─ BGE-M3 → Qdrant docjp_knowledge
  │              → filter theo research topic → prompt có citation
  └── Coding Agent (experimental, ENABLE_AGENT)
                 LangGraph: agent → MCP tools → agent

RAG prompt / SQL result
  ▼
LiteLLM Proxy (OpenAI-compatible)
  ├── local Qwen qua Ollama
  ├── cloud fallback (OpenAI/Azure/Grok tùy cấu hình)
  ▼
SSE token/status/meta/sources/done
  ▼
React ghép delta, hiển thị citation và preview trang PDF
```

### Câu nói mở đầu có thể dùng khi phỏng vấn

> “Meibook là chatbot nội bộ đa mode. Tôi không dùng LLM cho mọi việc: HR factual query được route deterministic vào SQLite; MES ưu tiên SQL template và chỉ fallback sang text-to-SQL có AST validation, read-only database và kiểm tra output; RAG dùng BGE-M3 multilingual với Qdrant cho tài liệu chính sách và research; còn LangGraph chỉ nằm ở coding agent thử nghiệm, không phải core MES/HR.”

Đây là điểm quan trọng: **LLM là một thành phần trong hệ thống, không phải toàn bộ hệ thống.** Routing, dữ liệu có cấu trúc, validation và observability quyết định độ tin cậy.

---

## 2. RAG: từ tài liệu đến câu trả lời

### 2.1 RAG là gì?

RAG (Retrieval-Augmented Generation) tách hai việc:

1. **Retrieval:** tìm các đoạn dữ liệu liên quan từ knowledge base.
2. **Generation:** đưa câu hỏi + các đoạn tìm được vào LLM để tạo câu trả lời.

RAG giải quyết một phần vấn đề knowledge cutoff và hallucination của LLM, nhưng **không tự động bảo đảm đúng**. Nếu chunk sai, retrieval sai hoặc prompt không ép model bám nguồn, câu trả lời vẫn sai.

Một pipeline chuẩn:

```text
raw documents
  → parse/OCR
  → chunk
  → embedding
  → vector database

user query
  → query embedding
  → top-k nearest-neighbor search
  → metadata/score filtering
  → prompt assembly
  → LLM answer + citations
```

Nguồn học chính thức:

- [Qdrant Search Concepts](https://qdrant.tech/documentation/concepts/search/)
- [Qdrant Filtering](https://qdrant.tech/documentation/concepts/filtering/)
- [Sentence Transformers semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)

### 2.2 Meibook triển khai RAG như thế nào?

Các file cần nhớ:

- [src/rag/rag_pipeline.py](../src/rag/rag_pipeline.py): orchestration query → retrieval → prompt → model → post-processing.
- [src/rag/vector_store.py](../src/rag/vector_store.py): Qdrant collection, payload filter, search.
- [src/rag/embedder.py](../src/rag/embedder.py): BGE-M3 embedding.
- [src/rag/parser.py](../src/rag/parser.py): parse/OCR/chunk.
- [src/rag/prompts.py](../src/rag/prompts.py): system prompt/context/citation.

Meibook có các collection độc lập:

| Collection | Dữ liệu | Cách giới hạn phạm vi |
|---|---|---|
| `mkac_knowledge` | HR, policy, company documents | intent/category + session cố định |
| `docjp_knowledge` | research documents, thường Nhật/Việt | `metadata.category` theo topic |
| `docmind_documents` | tài liệu upload theo session | `session_id` |

Tách collection giúp mỗi corpus có lifecycle, threshold và quyền truy cập riêng; tránh câu hỏi HR lấy nhầm chunk research.

### 2.3 Embedding và vector search

Meibook dùng `BAAI/bge-m3` qua SentenceTransformers:

- 1024 chiều.
- L2-normalized.
- cosine distance trong Qdrant.
- multilingual, nên query tiếng Nhật có thể embed trực tiếp với tài liệu tiếng Nhật.
- encode có lock và `torch.inference_mode()` để an toàn khi nhiều request dùng cùng model.

**Tại sao normalize + cosine?** Khi vector đã normalize, cosine similarity phản ánh hướng semantic của vector và ít bị ảnh hưởng bởi độ dài tuyệt đối. Đây là lựa chọn phổ biến cho semantic retrieval; quan trọng nhất là metric lúc index và lúc query phải nhất quán.

Meibook hiện dùng **pure dense retrieval**, chưa dùng hybrid dense+sparse/BM25. Trade-off:

- Ưu: kiến trúc đơn giản, multilingual semantic matching tốt.
- Nhược: có thể yếu với mã Lot, product ID, error code hoặc từ khóa phải khớp chính xác.
- Cách bù trong hệ thống: route các câu hỏi MES có mã/số liệu sang SQL deterministic thay vì bắt vector search xử lý.

### 2.4 Chunking: không có một kích thước đúng cho mọi tài liệu

Trong [src/rag/parser.py](../src/rag/parser.py), prose dùng:

- `CHUNK_SIZE = 1400` ký tự.
- `CHUNK_OVERLAP = 220` ký tự.
- overlap theo line, tránh cắt giữa dòng.
- line quá dài được hard-split.
- chunk dưới 20 ký tự bị bỏ.

Bảng lớn có strategy riêng:

- bảng từ 12 row trở lên được tách theo row.
- lặp header ở mỗi chunk.
- tối đa khoảng 15 row hoặc 3600 ký tự/chunk.
- compact các row sparse từ Excel bằng tên cột.

**Vì sao không cắt tất cả bằng character count?** Với bảng nhân sự hoặc approval matrix, cắt giữa record khiến giá trị bị trộn giữa các dòng và mất nghĩa cột. Lặp header khiến mỗi chunk tự chứa đủ schema cục bộ.

**Trade-off của overlap:**

- overlap giúp thông tin ở ranh giới chunk không bị mất.
- overlap quá lớn làm tăng số vector, chi phí embedding và duplicate context.
- chunk quá nhỏ mất ngữ cảnh; chunk quá lớn retrieval kém chính xác và prompt nhanh đầy.

### 2.5 Ingestion: parse, OCR, metadata, preview

Pipeline offline là:

1. Manifest-driven discovery: chỉ index tài liệu có trong manifest.
2. SHA-256 checksum: file không đổi được skip; `--reindex` mới ép xử lý lại.
3. Ưu tiên curated Markdown đã sửa tay nếu có.
4. PDF thử PyMuPDF native extraction.
5. Trang có dưới 80 ký tự native text mới fallback sang Docling/EasyOCR.
6. Chunk + enrich identity prefix trước embedding.
7. Upsert point vào Qdrant với payload.
8. Render mỗi trang PDF thành PNG 2× để citation preview.
9. Full run có thể prune tài liệu không còn trong manifest.

Identity prefix của chunk DocJP có thông tin như knowledge base, title, category và tên tổ chức bằng tiếng Nhật. Mục đích là giúp embedding giữ cả **nội dung đoạn** lẫn **ngữ cảnh tài liệu**.

Payload quan trọng:

```text
session_id
text
source_file
page_number
chunk_index
content_type
metadata.category
metadata.image_path
checksum
```

### 2.6 Retrieval filter và prompt assembly

Qdrant filter luôn có điều kiện `session_id`; research thêm `metadata.category`. Điều này là filter server-side, tốt hơn việc lấy toàn bộ top-k rồi lọc trong Python.

MKAC có thêm các bước sau retrieval:

1. query preprocessing: bỏ boilerplate tên công ty để tránh làm loãng semantic signal.
2. intent category filter: employee statistics → category nhân sự; company profile → category pháp lý/company.
3. relative score floor: bỏ result có score thấp hơn khoảng 85% score tốt nhất.
4. nếu không còn result → web/general fallback.

Meibook chưa có cross-encoder reranker riêng. Intent filter + relative score floor là một dạng post-retrieval heuristic nhẹ, đổi precision tiềm năng lấy latency đơn giản hơn.

Trong [src/rag/prompts.py](../src/rag/prompts.py):

- chọn system prompt theo mode/scope.
- format context thành các đoạn đánh số.
- truncate chunk trong prompt để tránh overflow.
- đưa conversation history giới hạn vào prompt.
- Research hiển thị filename/page citation.
- tối đa một số preview image được đưa vào model vision nếu cần.

### 2.7 Bilingual và chống hallucination

Research Nhật không đi qua chuỗi dịch Nhật → Việt → retrieval. BGE-M3 multilingual embed query gốc trực tiếp. Đây là lựa chọn tránh semantic drift do double translation.

Các lớp giảm hallucination:

- prompt yêu cầu chỉ trả lời trong scope tài liệu.
- nếu không có context thì nói không tìm thấy.
- citation được trả như structured metadata, không để model tự bịa link.
- local Qwen output được clean: bỏ `<think>`, image marker và repetitive loop.
- local answer rỗng/degraded có thể retry cloud.
- MES còn có output verification mạnh hơn, xem phần 4.

**Nhưng cần nói chính xác khi phỏng vấn:** RAG không “ngăn hallucination tuyệt đối”. Nó chỉ làm grounding tốt hơn; correctness vẫn cần retrieval evaluation, prompt constraint, output validation và human/business checks.

---

## 3. LLM routing, inference và streaming

### 3.1 LiteLLM làm gì?

LiteLLM Proxy cung cấp một OpenAI-compatible gateway trước nhiều backend model. Ứng dụng gọi một API shape thống nhất, còn proxy route tới local Ollama/Qwen hoặc provider cloud theo alias/fallback config.

Nguồn chính thức:

- [LiteLLM Proxy](https://docs.litellm.ai/docs/proxy/quick_start)
- [LiteLLM Fallbacks](https://docs.litellm.ai/docs/completion/fallback)
- [LiteLLM Streaming](https://docs.litellm.ai/docs/completion/stream)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)

Lợi ích kiến trúc:

- application không hard-code provider-specific SDK flow.
- đổi model/fallback bằng config.
- thống nhất timeout, retry, logging và streaming boundary.
- dễ dùng local model để tiết kiệm và cloud để reliability.

### 3.2 Fallback không chỉ là “thử model khác”

Trong Meibook, local Qwen là lựa chọn ưu tiên cho chi phí/latency; nếu output rỗng hoặc degraded sau clean-up thì có thể retry model cloud.

Khi giải thích trade-off:

- **Local:** riêng tư hơn, có thể rẻ hơn, kiểm soát latency; nhưng chất lượng/capacity và GPU resource hạn chế.
- **Cloud:** model mạnh và ổn định hơn; nhưng tốn tiền, phụ thuộc mạng và cần policy dữ liệu.
- **Fallback:** tăng availability nhưng có thể trả lời chậm hơn và làm request tốn hai lần inference.

Nên gắn model_used vào response/meta để debug và audit.

### 3.3 SSE trong Meibook

Backend dùng FastAPI `StreamingResponse`, không phải `EventSourceResponse`:

```text
data: {"type":"status", ...}\n\n
data: {"type":"sources", ...}\n\n
data: {"type":"meta", ...}\n\n
data: {"type":"token", "content":"..."}\n\n
data: {"type":"done"}\n\n
```

Frontend không dùng browser `EventSource`; frontend POST JSON qua `fetch`, đọc `ReadableStream`, dùng `TextDecoder` và buffer để xử lý trường hợp network chunk cắt giữa một SSE frame. `AbortController` cho phép hủy generation.

Các event chính:

| Event | Ý nghĩa |
|---|---|
| `status` | trạng thái localized: đang tìm kiếm, đang tạo câu trả lời… |
| `sources` | citation đến trước token |
| `meta` | model, mode, answer scope |
| `token` | append delta vào assistant message |
| `replace` | thay toàn bộ nội dung sau hậu xử lý |
| `error` | lỗi backend; frontend đưa vào placeholder |
| `done` | kết thúc stream |

Câu hỏi phỏng vấn thường gặp: **vì sao không dùng EventSource?** Vì request cần POST body JSON (`QueryRequest`) và cần AbortSignal; `fetch` streaming linh hoạt hơn GET-only EventSource.

Điểm cần tự nhận diện như một kỹ sư: nếu stream EOF trước event `done` mà không có error, client có thể giữ câu trả lời dở dang như thể hoàn tất. Đây là một improvement hợp lý: theo dõi `done`, báo incomplete stream nếu reader đóng sớm.

---

## 4. Agent, Text-to-SQL và LangGraph

### 4.1 MES: deterministic-first

MES không bắt đầu bằng “hãy để LLM suy nghĩ”. Route thực tế:

```text
intent detection
  → live MES API / snapshot DB
  → deterministic time SQL
  → deterministic highest-lot SQL
  → LLM SQL agent nếu cần
  → general MES answer
  → static unsupported fallback
```

HR structured Q&A còn deterministic hơn:

```text
normalize Unicode + bỏ dấu
  → keyword/regex intent
  → parameterized SQLite query
  → template answer
```

Đây là design rất tốt cho số liệu factual: câu hỏi headcount, department, lot, quantity không cần model tự suy diễn.

### 4.2 LLM SQL agent hoạt động thế nào?

LLM nhận `mes_semantic_model.json`, mô tả các view/cột/quan hệ/business rules, rồi phải trả JSON dạng:

```json
{"can_answer": true, "sql": "SELECT ...", "reason": "..."}
```

Nếu schema không trả lời được, model phải đặt `can_answer=false`.

SQL được bảo vệ theo defense-in-depth:

1. giới hạn độ dài.
2. chỉ một statement.
3. chỉ `SELECT` hoặc `WITH ... SELECT`.
4. parse bằng `sqlglot` AST.
5. cấm DDL/DML/PRAGMA/ATTACH.
6. chỉ allowlisted views.
7. allowlisted functions.
8. tự thêm LIMIT nếu thiếu.
9. SQLite mở `mode=ro`.
10. `PRAGMA query_only = ON`.
11. SQLite authorizer default-deny.
12. progress handler timeout + row limit.
13. output natural language phải chứa các ID/số liệu thực từ query result.

**Câu trả lời phỏng vấn tốt:** “Tôi không coi SQL do LLM sinh là trusted code. Tôi validate trước execution và validate lại câu trả lời sau execution against ground-truth rows.”

### 4.3 LangGraph: Meibook đang dùng ở đâu?

Meibook **có dùng LangGraph**, nhưng chỉ trong [src/agent/graph.py](../src/agent/graph.py) cho coding agent experimental, có `ENABLE_AGENT` gate. Nó không phải orchestration của core MES/HR.

Graph đơn giản:

```text
START → agent node
           │ tool_calls?
           ├── yes → tools node → agent node (loop)
           └── no  → END
```

- `StateGraph` giữ state.
- `agent` gọi LLM với bound tools.
- `ToolNode` chạy tool từ MCP adapters.
- conditional edge quyết định loop hay kết thúc.
- tools gồm filesystem/git/calendar tùy MCP server.

Đây là ReAct-style loop: model quyết định hành động, tool thực thi, model đọc observation rồi quyết định tiếp.

Nguồn chính thức:

- [LangGraph Overview](https://langchain-ai.github.io/langgraph/)
- [LangGraph Graph API](https://langchain-ai.github.io/langgraph/reference/graphs/)
- [LangGraph Tool Calling](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/)
- [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)

### 4.4 Plain Python hay LangGraph?

**Plain Python phù hợp khi:**

- số nhánh ít và biết trước.
- cần latency/predictability.
- business rule phải audit được.
- deterministic-first là ưu tiên.

**LangGraph phù hợp khi:**

- có loop agent/tool nhiều bước.
- state cần được model hóa rõ.
- cần checkpoint, resume, human-in-the-loop.
- conditional transitions tăng đến mức if/else thủ công khó bảo trì.

Không nên nói “LangGraph luôn tốt hơn”. Framework tạo abstraction, persistence và visualization nhưng thêm dependency, learning curve và overhead. Meibook dùng plain Python cho MES là lựa chọn có chủ ý, còn dùng LangGraph ở coding agent nơi tool loop tự nhiên hơn.

---

## 5. Các trade-off nên chủ động nói trong phỏng vấn

### 5.1 Dense-only retrieval

Hiện trạng: đơn giản, multilingual tốt. Rủi ro: exact code/ID yếu. Cải tiến có thể thử:

- hybrid dense + sparse/BM25.
- query expansion cho code/alias.
- deterministic route cho entity có cấu trúc.
- reranker cross-encoder sau top-k.

### 5.2 Heuristic score filtering thay reranker

`score_threshold` và relative floor nhanh, dễ debug nhưng score embedding không phải xác suất relevance. Cần benchmark trên dataset câu hỏi thật trước khi chọn threshold. Đo:

- Recall@k / Context Recall.
- MRR hoặc nDCG.
- Context Precision.
- answer faithfulness và answer relevancy.

### 5.3 Caching và token budget

Meibook adaptive `max_tokens`: câu đơn giản dùng ít token; research phức tạp dùng nhiều hơn. Đây là trade-off cost/latency/answer completeness. Đừng chỉ tối ưu token output; prompt context, history và image token cũng ảnh hưởng cost.

### 5.4 Citation architecture

Citation nên là data từ retrieval, không phải URL do model tự viết. Meibook lưu `source_file`, `page_number`, `score`, `image_path`; frontend mở preview ảnh trang gốc. Đây là cách nối grounded answer với UX kiểm chứng.

---

## 6. Bộ câu hỏi phỏng vấn và đáp án mẫu

### Q1. Hãy mô tả RAG pipeline của bạn.

**Đáp án mẫu:** “Tài liệu được manifest-discover, parse bằng curated Markdown/PyMuPDF/Docling OCR, chunk theo prose hoặc table-aware strategy, embed bằng BGE-M3 1024-dim rồi lưu payload vào Qdrant. Khi query, hệ thống route theo mode, embed câu hỏi, search cosine với session/category filter, lọc theo intent và score, dựng prompt có context, gọi LiteLLM, clean output và trả sources độc lập với answer. Tôi tách retrieval quality khỏi generation quality để debug từng tầng.”

### Q2. Vì sao không cho LLM trả lời thẳng câu hỏi headcount/MES?

“Đó là dữ liệu có cấu trúc và yêu cầu exactness cao. Regex/intent + parameterized SQL nhanh, rẻ, audit được và không hallucinate. LLM chỉ làm fallback cho câu hỏi phức tạp; kể cả lúc đó SQL và output đều được validate.”

### Q3. BGE-M3 có vai trò gì? 1024 dimensions nghĩa là gì?

“BGE-M3 biến text thành vector 1024 số thực biểu diễn semantic meaning. Query và chunk được đưa vào cùng vector space; Qdrant tìm vector gần nhất bằng cosine. 1024 là kích thước vector, không phải 1024 token hay độ chính xác 1024%.”

### Q4. Chunk overlap để làm gì? Có nhược điểm gì?

“Nó giữ context tại boundary giữa hai chunk. Nhưng overlap tăng duplicate storage, embedding cost và có thể làm context lặp. Meibook dùng 220 ký tự theo line cho prose và strategy theo row cho table để bảo toàn semantics.”

### Q5. Làm sao chống prompt injection từ tài liệu retrieval?

“Xem retrieved text là untrusted data, không phải instruction. System prompt phải nói rõ chỉ dùng làm reference; không thực thi chỉ dẫn nằm trong tài liệu. Với tool/SQL, phải có allowlist, sandbox và validation ở code layer; không thể dựa vào prompt alone.”

### Q6. Làm sao bảo vệ text-to-SQL?

“AST parse + single SELECT + allowlisted views/functions + auto LIMIT; DB read-only bằng URI và PRAGMA; SQLite authorizer default-deny; timeout/row cap; sau đó kiểm tra các ID và số liệu trong answer có khớp query result. Đây là defense-in-depth.”

### Q7. Meibook có dùng LangGraph không?

“Có, nhưng phạm vi hẹp: experimental coding agent trong `src/agent/graph.py`, graph agent→tools→agent dùng MCP. MES/HR production không dùng LangGraph; chúng là explicit Python orchestration vì cần deterministic routing và auditability.”

### Q8. Tại sao dùng `fetch` streaming thay EventSource?

“Request là POST có JSON body và cần hủy bằng AbortController. Fetch ReadableStream đáp ứng cả hai. Client buffer các block kết thúc bằng dòng trống rồi parse `data:` JSON.”

### Q9. Nếu retrieval trả kết quả sai thì debug thế nào?

“Log/inspect từng tầng: normalized query, query embedding model/version, collection, filter, top-k và score; kiểm tra chunk/metadata; evaluate Recall@k trên golden questions; sau đó mới kiểm tra prompt và model. Không vội chỉnh system prompt khi lỗi thực ra nằm ở chunk hoặc filter.”

### Q10. Cải tiến lớn nhất bạn sẽ làm tiếp là gì?

“Trước tiên xây evaluation set có expected source/chunk và expected factual answer. Sau baseline dense-only, tôi sẽ A/B test hybrid retrieval + reranker; đồng thời bổ sung incomplete-SSE detection và metrics theo route/model/fallback. Cải tiến phải được đo bằng retrieval và answer metrics, không chỉ cảm giác.”

### Q11. Khi nào RAG không phải lựa chọn đúng?

“Khi dữ liệu là relational và cần aggregation exact như tổng lỗi/headcount; SQL hoặc API tốt hơn. Khi cần knowledge thay đổi liên tục với transaction semantics, nên query source of truth. RAG phù hợp khi cần tìm thông tin trong corpus unstructured và đưa context vào generation.”

### Q12. Làm sao đo chất lượng RAG?

“Có hai nhóm: retrieval metrics như Recall@k, MRR, nDCG, Context Precision/Recall; generation metrics như faithfulness/groundedness, answer relevancy, correctness và citation accuracy. Kết hợp automated eval với một bộ human-labeled queries theo từng mode.”

---

## 7. Glossary siêu ngắn

| Thuật ngữ | Nghĩa thực dụng |
|---|---|
| LLM | Model sinh text dựa trên token/context |
| Embedding | Vector biểu diễn semantic của text |
| Chunk | Đoạn tài liệu được index độc lập |
| Top-k | k kết quả gần nhất trả về |
| Score threshold | ngưỡng loại kết quả quá yếu |
| Payload filter | lọc metadata trong vector DB |
| Reranker | model đánh giá lại relevance của top candidates |
| Dense retrieval | tìm theo vector semantic |
| Sparse retrieval | tìm theo token/keyword, ví dụ BM25 |
| Hybrid search | kết hợp dense và sparse |
| Grounding | buộc answer dựa vào evidence |
| Hallucination | model sinh claim không được evidence hỗ trợ |
| ReAct | reasoning loop xen kẽ action/tool và observation |
| Tool calling | model phát sinh lời gọi hàm có schema |
| StateGraph | graph nodes/edges quản lý state trong LangGraph |
| SSE | server đẩy event một chiều qua HTTP stream |
| Fallback | chuyển sang model/route khác khi route trước lỗi |
| AST | cây cú pháp; dùng để validate SQL trước chạy |

---

## 8. Lộ trình học cấp tốc cho tối nay

### 60 phút đầu — kể được hệ thống

- Vẽ lại sơ đồ ở phần 1.
- Đọc nhanh `rag_pipeline.py`, `vector_store.py`, `mes_query_service.py`, `src/agent/graph.py`.
- Tập nói “Meibook đang dùng” vs “concept so sánh”.

### 60 phút tiếp — RAG sâu

- Tự giải thích chunk size/overlap, cosine, top-k, threshold, metadata filter.
- Lấy một câu hỏi mẫu và trace bằng tay: query → collection → filter → prompt → answer.
- Nhớ limitation: dense-only, chưa có cross-encoder reranker.

### 45 phút — agent và safety

- Học thuộc 5-tier MES fallback.
- Học thuộc các lớp SQL defense.
- So sánh plain Python orchestration với LangGraph bằng use case.

### 30 phút — LLM system design

- Giải thích LiteLLM gateway, local/cloud fallback, token budget.
- Giải thích `fetch` + SSE + AbortController.
- Nêu một improvement có metrics, không nói chung chung.

### 15 phút cuối — mock interview

Tự trả lời 5 câu: “kiến trúc”, “RAG”, “hallucination”, “LangGraph”, “debug retrieval”. Mỗi câu trong 60–90 giây, luôn có: **quyết định → lý do → trade-off → cách đo**.

---

## 9. Những ngộ nhận cần tránh

- “Meibook dùng LangGraph cho toàn bộ chatbot.” → Sai; chỉ coding agent experimental.
- “RAG thì không hallucinate.” → Sai; RAG chỉ cung cấp evidence.
- “Embedding dimension càng lớn càng tốt.” → Sai; phải benchmark latency, memory và quality.
- “LLM sinh SQL là agent an toàn nếu prompt nói chỉ SELECT.” → Sai; phải validate ở code/database.
- “Cosine score là xác suất đúng.” → Sai; score dùng để rank/filter, cần calibrate bằng eval.
- “SSE là WebSocket.” → Sai; SSE một chiều server→client qua HTTP.
- “Top-k tăng thì recall luôn tốt hơn.” → Có thể tăng recall nhưng context precision và prompt noise có thể giảm.
- “Model mạnh hơn là giải pháp cho retrieval sai.” → Thường không; sửa chunk/filter/index trước.
- “Cloud fallback luôn tốt.” → Nó đổi reliability lấy cost, privacy và latency.

---

## 10. Nguồn chính thức để đọc thêm

- [BGE-M3 model card — BAAI](https://huggingface.co/BAAI/bge-m3)
- [Sentence Transformers semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [Qdrant Concepts: Search](https://qdrant.tech/documentation/concepts/search/)
- [Qdrant Concepts: Filtering](https://qdrant.tech/documentation/concepts/filtering/)
- [Qdrant Hybrid Queries](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [LiteLLM Proxy quick start](https://docs.litellm.ai/docs/proxy/quick_start)
- [LiteLLM fallback routing](https://docs.litellm.ai/docs/completion/fallback)
- [LiteLLM streaming](https://docs.litellm.ai/docs/completion/stream)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Graph API](https://langchain-ai.github.io/langgraph/reference/graphs/)
- [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [MDN Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

---

## Chốt lại trong một câu

> “Điểm tôi học được khi xây Meibook là AI Engineer không chỉ chọn model; họ thiết kế cả đường đi của dữ liệu, kiểm soát retrieval, giới hạn tool/SQL, theo dõi fallback và đo chất lượng end-to-end.”
