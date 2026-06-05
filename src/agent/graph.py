"""
src/agent/graph.py
------------------
Xây dựng đồ thị Agent lập trình bằng LangGraph (Plan-and-Execute ReAct loop).
Kết nối với LiteLLM Proxy làm LLM Backend (coding-model) và MCP Client làm Tools Backend.
"""

import os
import logging
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agent.mcp_client import get_mcp_tools

logger = logging.getLogger(__name__)


# 1. Định nghĩa trạng thái của Agent (Agent State)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Ta có thể thêm các trạng thái tùy chỉnh khác như:
    # plan: str
    # current_step: int


# 2. Định nghĩa hệ thống LLM thông qua LiteLLM Proxy
def get_llm():
    # LiteLLM Proxy chạy ở port 4000 trên Máy 2
    proxy_url = os.getenv("LITELLM_URL", "http://localhost:4000/v1")
    logger.info(f"Connecting Agent to LLM Proxy: {proxy_url}...")
    
    return ChatOpenAI(
        model="coding-model",
        openai_api_key=os.getenv("LITELLM_MASTER_KEY", "sk-local"),
        openai_api_base=proxy_url,
        temperature=0.2,
    )


# 3. Tạo các nút (Nodes) trong đồ thị
def call_model(state: AgentState):
    """
    Nút gọi mô hình LLM để quyết định hành động tiếp theo.
    """
    messages = state["messages"]
    
    # Chèn System Message để định hình phong cách hoạt động bằng Tiếng Việt
    system_prompt = SystemMessage(
        content=(
            "Bạn là một AI Coding Agent chuyên nghiệp, giao tiếp hoàn toàn bằng Tiếng Việt.\n"
            "Nhiệm vụ của bạn là lập trình, sửa lỗi và thực thi các câu lệnh theo yêu cầu của người dùng.\n"
            "Hãy luôn tuân thủ các quy tắc sau:\n"
            "1. Suy nghĩ chậm rãi, lập kế hoạch chi tiết trước khi gọi bất kỳ công cụ (tool) nào.\n"
            "2. Giải thích rõ ràng bạn đang chuẩn bị làm gì bằng Tiếng Việt trước khi thực thi.\n"
            "3. Sử dụng các công cụ filesystem để đọc/ghi file và terminal để chạy các lệnh xác minh.\n"
            "4. Khi hoàn thành công việc, hãy tóm tắt những thay đổi đã làm và kết quả chạy thử."
        )
    )
    
    # Gộp system prompt vào đầu danh sách tin nhắn
    llm = get_llm()
    tools = get_mcp_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke([system_prompt] + list(messages))
    return {"messages": [response]}


def should_continue(state: AgentState):
    """
    Quyết định xem có tiếp tục gọi tool hay kết thúc vòng lặp.
    """
    last_message = state["messages"][-1]
    # Nếu mô hình có yêu cầu gọi tool (tool_calls)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# 4. Xây dựng đồ thị LangGraph
def create_agent_graph():
    # Lấy danh sách công cụ
    tools = get_mcp_tools()
    tool_node = ToolNode(tools)
    
    # Khởi tạo StateGraph
    workflow = StateGraph(AgentState)
    
    # Định nghĩa các node
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Cấu hình các cạnh (edges)
    workflow.set_entry_point("agent")
    
    # Cạnh có điều kiện: quyết định đi tiếp tới node tools hay dừng lại (END)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # Sau khi chạy tool xong, quay trở lại model để phân tích kết quả và lên kế hoạch tiếp
    workflow.add_edge("tools", "agent")
    
    # Compile đồ thị
    return workflow.compile()


# Instance đồ thị sẵn sàng để sử dụng
agent_executor = create_agent_graph()
