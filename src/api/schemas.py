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


class EmployeeAuthRequest(BaseModel):
    employee_id: str


class EmployeeResponse(BaseModel):
    id: str
    name: str
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
