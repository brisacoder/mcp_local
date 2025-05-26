import random
from typing import List
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


class GraphConfig:
    """
    Configuration for the state graph, including LLM and tool integration.

    Attributes:
        llm_with_tools (Runnable): The LLM configured with tools.
        llm_with_structured (Runnable): The LLM configured for structured output.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1")
        self.llm_with_tools = None
        self.llm_with_structured = self.llm.with_structured_output(schema=Weather)
        self.mcp_tools = []
        self.thread_id = random.randint(1, 10)
        self.config: RunnableConfig = None

    def create_config(self) -> RunnableConfig:
        """
        Creates a RunnableConfig instance for the current configuration.

        Returns:
            RunnableConfig: A configuration object for the Runnable.
        """
        self.config = {
            "configurable": {
                "llm_with_tools": self.llm_with_tools,
                "llm_with_structured": self.llm_with_structured,
                "thread_id": self.thread_id,
                "mcp_tools": self.mcp_tools,
            }
        }
        return self.config

    def set_mcp_tools(self, mcp_tools):
        """
        Sets the mcp_tools attribute for the instance.

        Args:
            mcp_tools: The tools or configuration to be assigned to the mcp_tools attribute.
        """
        self.mcp_tools = mcp_tools

    def set_llm_with_tools(self, tools: List[BaseTool]) -> None:
        """
        Binds a list of tools to the current language model (LLM) and stores the resulting LLM-with-tools instance.

        Args:
            tools (List[BaseTool]): A list of tool instances to bind to the LLM.

        Returns:
            None
        """
        self.llm_with_tools = self.llm.bind_tools(tools)

    def get_llm_with_tools(self) -> Runnable:
        """
        Returns the language model with tools as a Runnable object.

        Returns:
            Runnable: The language model instance equipped with tools.
        """
        return self.llm_with_tools

    def get_llm_with_structured(self) -> Runnable:
        """
        Returns the LLM (Large Language Model) instance with structured output capabilities.

        Returns:
            Runnable: An instance of a Runnable object that represents the LLM with structured output enabled.
        """
        return self.llm_with_structured


graph_config = GraphConfig()
