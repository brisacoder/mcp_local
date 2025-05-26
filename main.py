import asyncio
import inspect
from typing import Annotated, Any, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langchain_core.tools.structured import StructuredTool
from graph_config import graph_config
from mcp_tools import get_mcp_tools
from models import ConversationTopic, Flights, Topic, Weather
from langchain_openai import ChatOpenAI

load_dotenv(override=True)  # Load environment variables from .env file
# In memory


class State(TypedDict):
    """
    Represents the state of the application.

    Attributes:
        graph_state (str): The current state of the graph.
        messages (list[AnyMessage]): A list of messages, with additional processing defined by 'add_messages'.
    """

    graph_state: str
    messages: Annotated[list[AnyMessage], add_messages]
    mcp_tools: List[dict[str, Any]]


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
    graph_config.set_mcp_tools(tools)
    return Command(goto="send_user_query_node")


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
    llm_with_tools: Runnable = graph_config.get_llm_with_tools()
    ai_resp = llm_with_tools.invoke(state["messages"])
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
    tools = graph_config.mcp_tools
    tools_by_name = {tool.name: tool for tool in tools}
    result = []

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.arun(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return Command(update={"messages": result}, goto="send_tool_result_to_llm")


# Nodes
def filter_messages_and_rerun_node(state: State) -> Command[Literal["tools"]]:
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][-1:]]
    return Command(update={"messages": delete_messages}, goto="tools")


def send_tool_result_to_llm(state: State, config) -> Command[Literal["filter_messages_and_rerun_node", END]]:
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
    llm_with_structured: Runnable = graph_config.get_llm_with_structured()
    # You may want to use tool_result in the update, for now just return a placeholder
    human_message = HumanMessage(
        content="""
        Parse the Microsoft Playwright output and provide the summary.
        """
    )
    ai_resp = llm_with_structured.invoke(state["messages"] + [human_message])
    if isinstance(ai_resp, Weather):
        if ai_resp.error != "":
            print(f"Error getting weather forecast: {ai_resp.error}")
            return Command(goto="filter_messages_and_rerun_node")
        forecast = ""
        for day in ai_resp.days:
            forecast += f"Day: {day.weekday}, Temperature: {day.temperature}°C, Condition: {day.condition}\n"
        ai_message = AIMessage(content=f"Forecast for NYC is:\n{forecast}.")
        return Command(update={"messages": ai_message}, goto=END)
    if isinstance(ai_resp, Flights):
        flight_info = ""
        for flight in ai_resp.flights:
            flight_info += (
                f"Flight from {flight.departure_city} to {flight.arrival_city}, "
                f"Airline: {flight.airline}, Price: {flight.price}\n"
            )
        ai_message = AIMessage(content=f"Flight information:\n{flight_info}.")
        return Command(update={"messages": ai_message}, goto=END)
    raise ValueError(
        "The response from the LLM is not of type Weather. "
        "Expected a Weather object with structured output."
    )


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
    builder.add_node("filter_messages_and_rerun_node", filter_messages_and_rerun_node)
    builder.add_node("send_tool_result_to_llm", send_tool_result_to_llm)
    builder.add_node(ToolNode(tools))
    builder.add_edge("tools", "send_tool_result_to_llm")
    builder.add_edge(START, "load_mcp_tools_node")
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    # graph = builder.compile()
    with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    return graph


async def get_conversation_topic():
    """
    Asynchronously prompts the user for their travel needs, sends the input to a language model to determine the conversation topic, and returns both the user's message and the identified topic.

    Returns:
        tuple: A tuple containing the user's input message (str) and the determined conversation topic (str).
    """
    user_msg = input("Tell me about your travel needs? \n")

    llm = ChatOpenAI(model="gpt-4.1")
    llm_with_structured = llm.with_structured_output(schema=ConversationTopic)
    messages = [
        SystemMessage(content="Determine the topic of the conversation"),
        HumanMessage(content=user_msg),
    ]
    conversation: Conversation = await llm_with_structured.ainvoke(messages)
    return user_msg, conversation.topic.value


async def main():
    """
    Asynchronously initializes language model tools, binds them to a GPT-4.1 chat model, constructs a processing graph, and invokes the graph with a system message to obtain a weather-related response.

    Returns:
        None

    Side Effects:
        Prints the weather response to the console.
    """
    while True:
        user_msg, topic = await get_conversation_topic()
        if topic == Topic.OTHER.value:
            print("I cannot answer that question!")
        else:
            break
    if topic == Topic.WEATHER.value:
        graph_config.set_llm_with_structured(schema=Weather)
    else:
        graph_config.set_llm_with_structured(schema=Flights)
    
    graph = await build_graph()
    gc = graph_config.create_config()
    messages = [
        SystemMessage(
            content="You are a helpful assistant and can use browser tools to fullfill user's requests"
        ),
        HumanMessage(
            content=user_msg
        ),
    ]

    weather_response = await graph.ainvoke(
        {"messages": messages},
        config=gc,
    )
    for m in weather_response["messages"]:
        m.pretty_print()

    messages = [
        HumanMessage(
            content="Give me options for a flight from SF to NYC tomorrow, returning 3 days later"
        )
    ]
    flight_response = await graph.ainvoke({"messages": messages}, config=gc)
    for m in flight_response["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    # Run the main function in an asyncio event loop
    # This is necessary to run async functions in Python
    # especially when using libraries like langchain that require async execution.
    asyncio.run(main())
