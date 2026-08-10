"""
Agent 状态定义 - 带上下文管理
"""
from typing import TypedDict, Optional, List
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    # 输入
    user_input: str
    session_id: str
    intent: str  # "explain" | "quiz" | "summary" | "chat" | "unknown"

    # RAG 检索
    retrieved_docs: List[str]
    rephrased_query: str
    rag_relevance_scores: List[float]
    rag_context_compressed: bool
    rag_deduplicated: bool

    # 上下文管理
    context_summary: str
    context_token_count: int
    context_truncated: bool

    # 跨轮次追踪
    current_topics: List[str]
    subject_history: List[str]
    knowledge_gaps: List[str]
    conversation_summary: str
    turn_count: int

    # 输出
    answer: str
    mindmap_mermaid: str
