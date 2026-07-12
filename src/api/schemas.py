"""Pydantic request/response schemas cho API Gateway.

Tách khỏi ``main.py`` để hợp đồng dữ liệu của API nằm gọn một chỗ, dễ tra cứu
khi thêm/sửa endpoint. Giữ nguyên định nghĩa so với bản gốc.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    session_id: str
    message: str


class QueryRequest(BaseModel):
    session_id: str
    question: str
    stream: bool = True
    model: Literal["auto", "local", "openai", "grok"] = "auto"
    mode: Literal["mkac", "mes", "research"] = "mkac"
    ui_language: Literal["vi", "ja"] = "vi"
    employee_id: Optional[str] = None
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


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    model: str
    mode: str
    answer_scope: str


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
