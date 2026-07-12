import os
os.environ["ENABLE_AGENT"] = "true"
from src.agent.mcp_client import get_mcp_tools
from langchain_core.utils.function_calling import convert_to_openai_tool
import json

tools = get_mcp_tools()
print("Loaded:", [t.name for t in tools])
oai_tools = [convert_to_openai_tool(t) for t in tools]
with open("/app/scripts/mcp_tools.json", "w") as f:
    json.dump(oai_tools, f)
