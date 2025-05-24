import json
import asyncio
from typing import Dict
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

load_dotenv()


def load_servers(config_path: str) -> Dict:
    """Load server configuration from a JSON file (typically mcp_config.json)
    and creates an instance of each server (no active connection until 'start' though).

    Args:
        config_path: Path to the JSON configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    return config


async def get_tools() -> list[BaseTool]:
    """Main function to run the MCP client."""
    mcp_servers = load_servers("mcp_config.json")
    print("Loaded servers:", mcp_servers)

    # if platform.system() == 'Windows':
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    client = MultiServerMCPClient(mcp_servers["mcpServers"])

    tools = await client.get_tools()
    return tools


async def main():
    tools = await get_tools()
    agent = create_react_agent("openai:gpt-4.1", tools)
    weather_response = await agent.ainvoke({"messages": "Use a browser to find what is the weather in nyc?"})
    print("Weather Response:", weather_response)


if __name__ == "__main__":
    # Run the main function in an asyncio event loop
    # This is necessary to run async functions in Python
    # especially when using libraries like langchain that require async execution.
    asyncio.run(main())
