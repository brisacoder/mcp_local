import asyncio
import json
from typing import Dict, List

import aiofiles
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools.structured import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode


async def load_servers(config_path: str) -> Dict:
    """Load server configuration from a JSON file (typically mcp_config.json)
    and creates an instance of each server (no active connection until 'start' though).

    Args:
        config_path: Path to the JSON configuration file.
    """
    async with aiofiles.open(config_path, "r", encoding="utf-8") as config_file:
        content = await config_file.read()
        config = json.loads(content)

    return config


async def get_mcp_tools() -> List[StructuredTool]:
    """Main function to run the MCP client."""
    mcp_servers = await load_servers("mcp_config.json")

    client = MultiServerMCPClient(mcp_servers["mcpServers"])
    all_tools = await client.get_tools()
    return all_tools


async def main():
    load_dotenv(override=True)
    tools = await get_mcp_tools()
    tool_node = ToolNode(tools)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(content="You are a helpful travel assistant"),
        HumanMessage(content="Use the browser to find a flight from SFO to JFK"),
    ]
    ai_resp = await llm_with_tools.ainvoke(messages)
    ai_resp.pretty_print()
    ret = await tool_node.ainvoke({"messages": [ai_resp]})
    

if __name__ == "__main__":
    asyncio.run(main())
