"""
src/agent/mcp_client.py
-----------------------
Client kết nối tới các MCP Server (stdio) sử dụng langchain-mcp-adapters.
Tự động khởi tạo các tools (filesystem, git, terminal) để cung cấp cho LangGraph Agent.
"""

import os
import logging
from typing import List
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Thử import adapter của LangChain
try:
    from langchain_mcp_adapters.tools import load_mcp_tools
    HAS_MCP_ADAPTER = True
except ImportError:
    HAS_MCP_ADAPTER = False
    logger.warning("langchain-mcp-adapters is not installed. Fallback to native mock/local tools.")


def get_mcp_tools() -> List[BaseTool]:
    """
    Khởi tạo và tải tất cả các công cụ từ các MCP Server qua stdio.
    """
    tools = []
    workspace_dir = os.getenv("WORKSPACE_DIR", "/home/jkl0909/Code")

    if not HAS_MCP_ADAPTER:
        logger.warning("Using mock tools because langchain-mcp-adapters is not available.")
        # Trả về các mock tools cơ bản để chạy thử nếu chưa cài đặt adapter
        return _get_fallback_local_tools()

    # 1. Tải Filesystem MCP Tools
    try:
        logger.info(f"Connecting to filesystem MCP server for workspace: {workspace_dir}...")
        fs_tools = load_mcp_tools(
            "stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", workspace_dir]
        )
        tools.extend(fs_tools)
        logger.info(f"Loaded {len(fs_tools)} filesystem tools successfully.")
    except Exception as e:
        logger.error(f"Error loading filesystem MCP tools: {e}")

    # 2. Tải Git MCP Tools
    try:
        logger.info("Connecting to git MCP server...")
        git_tools = load_mcp_tools(
            "stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-git", "--repository", workspace_dir]
        )
        tools.extend(git_tools)
        logger.info(f"Loaded {len(git_tools)} git tools successfully.")
    except Exception as e:
        logger.error(f"Error loading git MCP tools: {e}")

    # Nếu không tải được công cụ nào, sử dụng fallback tools cục bộ
    if not tools:
        logger.warning("No MCP tools loaded. Fallback to local tools.")
        return _get_fallback_local_tools()

    return tools


def _get_fallback_local_tools() -> List[BaseTool]:
    """
    Các tool dự phòng bằng Python nguyên bản khi không kết nối được MCP Server.
    Đảm bảo Agent luôn có các tool cơ bản để đọc ghi file.
    """
    from langchain_core.tools import tool

    @tool
    def read_file(file_path: str) -> str:
        """Đọc nội dung của một tệp tại đường dẫn được cung cấp."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Ghi nội dung mới vào tệp tại đường dẫn được cung cấp (ghi đè)."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    @tool
    def list_dir(directory: str = ".") -> str:
        """Liệt kê các tệp và thư mục con trong thư mục chỉ định."""
        try:
            items = os.listdir(directory)
            return "\n".join(items)
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    return [read_file, write_file, list_dir]
