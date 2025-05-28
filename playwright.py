import asyncio
import json
import re
import yaml
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
from langchain_core.prompts import ChatPromptTemplate


def extract_and_parse_yaml(text_content_list):
    """
    Given a list of TextContent objects (with `.text` attribute containing YAML blocks and fences),
    this function extracts the YAML portion between ```yaml fences and parses it into native Python.
    """
    # Combine all text parts into one string
    combined_text = "".join(part.text for part in text_content_list)

    # Extract the YAML block between ```yaml and the next ```
    match = re.search(r"```yaml\s*([\s\S]+?)```", combined_text)
    if not match:
        raise ValueError("No YAML block found in the provided content.")

    yaml_blob = match.group(1)

    # Parse the YAML blob into Python data structures
    tree = yaml.safe_load(yaml_blob)
    return tree


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
        llm = ChatOpenAI(model="gpt-4.1", temperature=0.1)
        llm_with_tools = llm.bind_tools(tools)

        nav_res = await session.call_tool(
            "browser_navigate", {"url": "https://www.google.com/travel/flights"}
        )
        snap_res = await session.call_tool("browser_snapshot", {})
        snapshot_tree = extract_and_parse_yaml(snap_res.content)
        # snapshot_res = await session.call_tool("browser_snapshot", {})

        messages = [
            SystemMessage(
                content=(
                    "You are a helpful travel assistant and an expert on parsing Microsoft "
                    "PLaywright accessibility tree"
                )
            ),
            (
                "human",
                "Use the Microsoft Playwright accessibility tree snapshot for "
                "https://www.google.com/travel/flights below and tools available "
                "to take all "
                "actions to the search a flights from SFO to JFK. Think about all the actions "
                "needed provide the complete list of tool calls "
                "to achieve the goal."
                "\n"
                "You can take any tool action on the webpage necessary to search for flights, "
                "such as, but not limited to, click, type, drag, etc. If you are missing user "
                "preferences such as departure time, make a best guess, do not ask user "
                "for further information "
                ""
                "Assume the browser is already open on the webpage"
                "\n"
                "Microsoft Playwright tree:"
                "\n"
                "{tree}",
            ),
        ]

        chat_template = ChatPromptTemplate.from_messages(messages)
        messages_from_template = chat_template.format_messages(
            tree=json.dumps(snapshot_tree)
        )

        ai_resp = await llm_with_tools.ainvoke(messages_from_template)
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
                    "Use the Microsoft Playwright accessibility webpage snapshot and tools available "
                    "to take all further "
                    "actions to complete the search in the next step. Think about all the actions "
                    "needed to search for flights and provide the complete list of tool calls "
                    "to achieve the goal."
                    ""
                    "You can take any tool action on the webpage necessary to search for flights, "
                    "such as, but not limited to, click, type, drag, etc. If you are missing user "
                    "preferences such as departure time, make a best guess, do not ask user "
                    "for further information"
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
