"""Pydantic request/response schemas cho API Gateway.

Tách khỏi ``main.py`` để hợp đồng dữ liệu của API nằm gọn một chỗ, dễ tra cứu
khi thêm/sửa endpoint. Giữ nguyên định nghĩa so với bản gốc.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    session_id: str
    message: str


class QueryRequest(BaseModel):
    session_id: str
    question: str
    stream: bool = True
    model: Literal["auto", "local", "openai"] = "auto"
    mode: Literal[
        "mkac", "mes", "wms", "cctvai", "research", "qa"
    ] = "mkac"
    ui_language: Literal["vi", "ja"] = "vi"
    employee_id: Optional[str] = None
    # Optional UI hint for a server-prepared WMS suggestion. The API validates
    # the ID again against mode, language and canonical question.
    quick_answer_id: Optional[str] = None
    conversation_context: List[Dict[str, Any]] = Field(default_factory=list)
    # Research corpus selector. ``topic`` queries the shared DocJP collection;
    # ``upload`` queries documents isolated by the request session UUID.
    # None preserves compatibility with older clients: a supplied topic means
    # ``topic`` scope, otherwise the request uses the upload session.
    research_scope: Optional[Literal["topic", "upload"]] = None
    # Research knowledge scope: one of the predefined topic ids in
    # config/research_topics.json, or "all" for the whole DocJP corpus.
    # None keeps the legacy per-session document behaviour.
    research_topic: Optional[str] = None
    # QA only: start a new follow-up thread (the UI sends true when the QA
    # conversation is empty). The carried agent context lives server-side.
    qa_reset_context: bool = False


class WmsDatasetEvidence(BaseModel):
    model_config = {"extra": "ignore"}

    dataset: str = ""
    status: str = ""
    reason_code: str = ""
    source_state: str = ""
    candidate_row_count: int = 0
    inserted_row_count: int = 0
    invalid_quantity_row_count: int = 0
    source_as_of: str = ""
    source_as_of_state: str = ""
    source_as_of_basis: str = ""
    source_timezone: str = "unverified"
    semantic_epoch: str = ""


class WmsPagination(BaseModel):
    page: int
    page_size: int
    total_count: int
    has_more: bool


class WmsAnswerMetadata(BaseModel):
    model_config = {"extra": "ignore"}

    contract_version: str = ""
    data_contract_version: str = ""
    semantic_contract_version: str = ""
    intent: str = ""
    domain: str = ""
    status: str = ""
    reason_codes: List[str] = Field(default_factory=list)
    imported_at: str = ""
    source_as_of: str = ""
    source_as_of_state: str = ""
    source_as_of_basis: str = ""
    source_timezone: str = "unverified"
    semantic_epoch: str = ""
    dataset_evidence: List[WmsDatasetEvidence] = Field(default_factory=list)
    grain: str = ""
    pagination: Optional[WmsPagination] = None


class CctvaiAnswerMetadata(BaseModel):
    model_config = {"extra": "ignore"}

    intent: str = ""
    domain: str = ""
    status: str = ""
    reason_codes: List[str] = Field(default_factory=list)
    latest_event_at: str = ""
    replica_lag_state: str = ""
    grain: str = ""
    schema_version: str = ""
    data_contract_version: str = ""
    semantic_contract_version: str = ""
    source_system: str = ""


# QA (Agent QMS 5M1E) citations are SQL/view/record references, not PDF pages,
# so they never go through the RAG source formatter or /sources/preview. The
# adapter in src/integrations/qa_agent_service.py has already allowlisted and
# capped every field; raw SQL from the agent is never part of this contract.
QaKeyValue = Union[int, float, str]


class QaCitationChunk(BaseModel):
    model_config = {"extra": "ignore"}

    id: Optional[Union[int, str]] = None
    source_view: str = ""
    source_key: Optional[Union[int, str]] = None
    inspection_key: Optional[Union[int, str]] = None


class QaCitation(BaseModel):
    model_config = {"extra": "ignore"}

    type: Literal["sql", "row", "semantic"]
    views: List[str] = Field(default_factory=list)
    keys: Dict[str, List[QaKeyValue]] = Field(default_factory=dict)
    inspection_keys: List[Union[int, str]] = Field(default_factory=list)
    chunks: List[QaCitationChunk] = Field(default_factory=list)


class QaAnswerMetadata(BaseModel):
    model_config = {"extra": "ignore"}

    provider: str = "qms_5m1e_agent"
    # The agent answers in Vietnamese only. Follow-up context (API 1.1) is kept
    # server-side per session; only these flags reach the UI, never the raw
    # context/history the agent returns.
    answer_language: Literal["vi"] = "vi"
    context_supported: bool = False
    # A previous QA answer's context was sent with this question.
    context_applied: bool = False
    # The agent asked for more detail instead of answering.
    clarification: bool = False
    # The question as the agent understood it after merging context.
    effective_question: str = ""
    client_ref: str = ""
    elapsed_ms: Optional[float] = None
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    citations: List[QaCitation] = Field(default_factory=list)
    citations_truncated: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    model: str
    mode: str
    answer_scope: str
    wms_metadata: Optional[WmsAnswerMetadata] = None
    cctvai_metadata: Optional[CctvaiAnswerMetadata] = None
    qa_metadata: Optional[QaAnswerMetadata] = None
    artifact: Optional[Dict[str, Any]] = None


class SessionInfoResponse(BaseModel):
    session_id: str
    num_chunks: int
    files: List[str]
    num_files: int


class ResearchDemoResponse(BaseModel):
    enabled: bool
    ready: bool
    session_id: str
    num_chunks: int
    num_files: int
    files: List[str]
    source_files: List[str]


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
    collection: str = ""
    session_id: str = ""
    default_topic: str = ""
    allow_all: bool = True
    topics: List[ResearchTopic] = Field(default_factory=list)


class EmployeeAuthRequest(BaseModel):
    employee_id: str


class EmployeeResponse(BaseModel):
    id: str
    name: str
    company_email: str = ""
    gender: str = ""
    position: str = ""
    department: str = ""
    greeting: str = ""
    department_size: int = 0
    department_heads: List[str] = Field(default_factory=list)
    department_deputies: List[str] = Field(default_factory=list)


class EmployeeAuthResponse(BaseModel):
    employee: EmployeeResponse


class AgentRequest(BaseModel):
    session_id: str
    task: str


class AgentResponse(BaseModel):
    session_id: str
    status: str
    output: str
    steps: List[Dict[str, Any]]
