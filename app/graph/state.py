"""
Состояние LangGraph агента.

TypedDict описывает данные, которые передаются между узлами графа.
"""

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Состояние графа ИИ-ассистента.

    Attributes:
        messages: История сообщений чата. Аннотация `add_messages`
                  автоматически мержит новые сообщения в список.
        user_id:  Идентификатор пользователя (для разделения памяти).
        memory_context: Контекст, извлечённый из долгосрочной памяти (Qdrant)
                        перед вызовом LLM. Подклеивается к system prompt.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    memory_context: str
    model: str
