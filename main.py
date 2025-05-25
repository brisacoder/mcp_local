import asyncio
import inspect
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, AnyMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from graph_config import Weather, graph_config
from mcp_tools import get_mcp_tools

load_dotenv(override=True)  # Load environment variables from .env file


class State(TypedDict):
    """
    Represents the state of the application.

    Attributes:
        graph_state (str): The current state of the graph.
        messages (list[AnyMessage]): A list of messages, with additional processing defined by 'add_messages'.
    """

    graph_state: str
    messages: Annotated[list[AnyMessage], add_messages]
    mcp_tools: list[BaseTool]


async def load_mcp_tools_node(state: State) -> Command[Literal["send_user_query_node"]]:
    """
    Loads the MCP tools into the state and transitions to the user query node.

    This function retrieves the MCP tools from the configuration, updates the state with these tools,
    and prepares to send a user query.

    Args:
        state (State): The current state of the application.
        config: Configuration dictionary that may include tool settings.

    Returns:
        Command[Literal["send_user_query_node"]]: A command object that updates the state with tools
        and transitions to the "send_user_query_node".
    """
    frame = inspect.currentframe()
    if frame is not None:
        print(frame.f_code.co_name)
    else:
        print("Could not get current frame name")
    tools = await get_mcp_tools()
    return Command(update={"mcp_tools": tools}, goto="send_user_query_node")


def send_user_query_node(state, config) -> Command[Literal["tools", END]]:
    """
    Executes a tool-calling node that interacts with an LLM
    configured in the state to answer a weather-related query.

    This function retrieves the LLM with tool capabilities from the provided state configuration, constructs a
    message asking for the weather in NYC, and invokes the LLM. The response is wrapped in a Command object to
    update the conversation and transition to the "end_node".

    Args:
        state (dict): The current state containing configuration and other context.

    Returns:
        Command[Literal["end_node"]]: A command object
        containing the AI's response and the next node to transition to.

    Raises:
        ValueError: If the LLM with tools is not configured in the state.
    """
    frame = inspect.currentframe()
    if frame is not None:
        print(frame.f_code.co_name)
    else:
        print("Could not get current frame name")
    llm_with_tools: Runnable = config["configurable"]["llm_with_tools"]
    messages = [
        HumanMessage(
            content="Use a browser to find what is the weather in NYC and provide a summary"
        )
    ]
    ai_resp = llm_with_tools.invoke(state["messages"] + messages)
    if ai_resp.tool_calls:
        # If the AI response contains tool calls, we can handle them here
        return Command(update={"messages": ai_resp}, goto="tools")
    return Command(update={"messages": ai_resp}, goto=END)


def tools_node(state: State, config) -> Command[Literal["send_tool_result_to_llm"]]:
    """
    Handles the execution of tools based on the AI's response in the state.

    This function processes the tool calls from the AI response, executes them, and prepares to send the results
    back to the LLM for further processing.

    Args:
        state (State): The current state containing messages and other context.

    Returns:
        Command[Literal["send_tool_result_to_llm"]]: A command object that updates the messages with the tool results
        and transitions to the "send_tool_result_to_llm" node.

    Notes:
        - Assumes that the latest message in `state["messages"]` contains tool calls to be executed.
    """
    last_message = state["messages"][-1]
    # Only AIMessage is expected to have tool_calls
    if not hasattr(last_message, "tool_calls") or not getattr(
        last_message, "tool_calls", None
    ):
        raise ValueError(
            "No tool calls found in the latest message or message type does not support tool calls."
        )

    tools = config["configurable"]["tools"]
    tools_by_name = {tool.name: tool for tool in tools}
    result = []

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.arun(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return Command(update={"messages": result}, goto="send_tool_result_to_llm")


def send_tool_result_to_llm(state: State, config) -> Command[Literal[END]]:
    """
    Sends the result of a tool execution to an LLM (Large Language Model) for further processing.

    Args:
        state (State): The current state containing messages and other context.
        config: Configuration dictionary that may include LLM and tool integration settings.

    Returns:
        Command[Literal["end_node"]]: A command object that updates the messages with the latest tool result and transitions to the "end_node".

    Raises:
        ValueError: If the LLM with tools is not configured in the provided config.

    Notes:
        - Assumes the latest message in `state["messages"]` is the tool result to be sent.
        - The function currently returns a placeholder update with the tool result.
    """
    llm_with_structured: Runnable = config["configurable"]["llm_with_structured"]
    # You may want to use tool_result in the update, for now just return a placeholder
    human_message = HumanMessage(
        content="Provide a summary of the weather in NYC based on the Microsoft Playwright output"
    )
    weather: Weather = llm_with_structured.invoke(state["messages"] + [human_message])
    forecast = ""
    for day in weather.days:
        forecast += f"Day: {day.weekday}, Temperature: {day.temperature}°C, Condition: {day.condition}\n"
    ai_resp = AIMessage(
        content=f"Forecast for NYC is:\n{forecast}."
    )
    return Command(update={"messages": ai_resp}, goto=END)


async def build_graph():
    """
    Builds and compiles a state graph with predefined nodes and edges, then displays its visualization.

    Returns:
        StateGraph: The compiled state graph object.
    """
    tools = await get_mcp_tools()
    graph_config.set_llm_with_tools(tools)
    builder = StateGraph(State)
    builder.add_node("load_mcp_tools_node", load_mcp_tools_node)
    builder.add_node("send_user_query_node", send_user_query_node)
    builder.add_node("send_tool_result_to_llm", send_tool_result_to_llm)
    builder.add_node(ToolNode(tools))
    builder.add_edge("tools", "send_tool_result_to_llm")
    builder.add_edge(START, "load_mcp_tools_node")
    graph = builder.compile()
    with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    return graph


async def main():
    """
    Asynchronously initializes language model tools, binds them to a GPT-4.1 chat model, constructs a processing graph, and invokes the graph with a system message to obtain a weather-related response.

    Returns:
        None

    Side Effects:
        Prints the weather response to the console.
    """
    graph = await build_graph()
    weather_response = await graph.ainvoke(
        {"messages": [SystemMessage(content="You are a helpful assistant.")]},
        config=graph_config.create_config(),
    )
    for m in weather_response["messages"]:
        m.pretty_print()

if __name__ == "__main__":
    # Run the main function in an asyncio event loop
    # This is necessary to run async functions in Python
    # especially when using libraries like langchain that require async execution.
    asyncio.run(main())
