from typing import List
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class WeatherDay(BaseModel):
    """A model to represent weather data for a single day."""

    temperature: float = Field(description="The weather temperature")
    condition: str = Field(
        description="The weather condition, e.g., sunny, rainy, etc."
    )
    weekday: str = Field(description="The day of the week, e.g., Monday, Tuesday, etc.")


class Weather(BaseModel):
    """A model to represent a list of weather days."""

    days: List[WeatherDay] = Field(
        description="A list of weather data for multiple days"
    )


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
                "thread_id": "1",
            }
        }
        return self.config

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
