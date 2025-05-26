import json
from typing import Dict, List

import aiofiles
from langchain_core.tools import BaseTool
from langchain_core.tools.structured import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient


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
