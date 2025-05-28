import asyncio
import json
from typing import Dict, List

import aiofiles
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools.structured import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
from langchain_openai import ChatOpenAI
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
    """
    Asynchronously initializes environment variables, loads MCP tools, and sets up a language model with tool bindings.
    Simulates a conversation with a travel assistant to find a flight from SFO to JFK using browser tools.
    Prints the AI's response and invokes the tool node with the AI's output.

    Steps:
    1. Loads environment variables with override enabled.
    2. Retrieves and initializes MCP tools.
    3. Sets up a GPT-4o language model with specified temperature.
    4. Binds the tools to the language model.
    5. Constructs a conversation with system and human messages.
    6. Invokes the language model with the conversation and prints the response.
    7. Passes the AI's response to the tool node for further processing.

    Returns:
        None
    """
    # Load OpenAI key
    load_dotenv(override=True)

    # tools = await get_mcp_tools()

    mcp_servers = await load_servers("mcp_config.json")

    client = MultiServerMCPClient(mcp_servers["mcpServers"])

    async with client.session("playwright") as session:

        tools = await load_mcp_tools(session)
        tool_node = ToolNode(tools)
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        llm_with_tools = llm.bind_tools(tools)
        messages = [
            SystemMessage(content="You are a helpful travel assistant"),
            HumanMessage(content=("Use a browser to find a flight from SFO to JFK")),
        ]
        ai_resp = await llm_with_tools.ainvoke(messages)
        ai_resp.pretty_print()
        messages += [ai_resp]

        # tool_output has the page snapshot
        tool_output = await tool_node.ainvoke({"messages": [ai_resp]})
        # The browser windows is closed by now. Usually, Gooogle Flights
        messages += tool_output["messages"]
        # This will print the tool output, which is a snapshot of the browser page. Lots of information
        # print(f"Tool output: {tool_output}")

        # Now let's try with more instructions as a possible work around for the browser being closed
        messages += [
            HumanMessage(
                content=(
                    "Type in all the necessary information on the search form and provide a list of flights. If you lack "
                    "user preferences such as departure time, make a best guess, do not ask user for further information"
                )
            )
        ]
        ai_resp = await llm_with_tools.ainvoke(messages)
        # This will sometimes generate an error such as "It seems there was an issue with navigating
        # to the Google Flights page. Let me attempt to open a different website to search for
        # flights from SFO to JFK. I'll try using Expedia for this purpose."

        # Alternatively it might have two tool calls as such as browser_click or browser_type.
        # But the point is that the browser is closed and there is no snapshot context.
        # Rinse and repeat
        ai_resp.pretty_print()
        # IF there are tool calls in the response. These need to be sent as a batch to
        # playwright/browser. Not in parallel, and not one at a time.
        messages += [ai_resp]
        tool_output = await tool_node.ainvoke({"messages": [ai_resp]})
        tool_output["messages"].pretty_print()
        # Since there is no smapshot available, the error normally is:
        # ""Error: ToolException('Error: No current snapshot available. Capture a
        # snapshot of navigate to a new location first.')\n Please fix your mistakes.""


if __name__ == "__main__":
    asyncio.run(main())
