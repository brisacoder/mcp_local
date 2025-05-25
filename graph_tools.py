import asyncio
import json
from typing import Dict

import aiofiles
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class GraphTools:
    """
    GraphTools is a utility class for managing and operating on a collection of tools within a graph-based context.

    Attributes:
        tools (list[BaseTool]): A list of tool instances used by the graph.

    Methods:
        __init__(tools: list[BaseTool]):
            Initializes the GraphTools instance with a provided list of tools.

        create():
            Asynchronously creates and returns a GraphTools instance with tools loaded via the get_tools() coroutine.
    """

    def __init__(self, tools: list[BaseTool]):
        """
        Initializes the GraphTools class with a list of tools.

        Args:
            tools (list[BaseTool]): A list of tools to be used in the graph.
        """
        self.tools = tools

    @classmethod
    async def create(cls):
        """
        Asynchronously creates an instance of the class with initialized tools.

        Returns:
            An instance of the class with tools loaded asynchronously.
        """
        tools = await get_mcp_tools()
        return cls(tools)
