"""
LangGraph 状态图构建 - 带上下文管理
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState
from agent.nodes import (
    query_node, context_manager_node, rag_node,
    chat_node, thought_node
)

# 短期记忆：每个 session_id 对应独立的历史记录
memory = MemorySaver()


def build_graph():
    """构建并编译 Agent 状态图"""
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("query", query_node)
    graph.add_node("context_manager", context_manager_node)
    graph.add_node("rag", rag_node)
    graph.add_node("chat", chat_node)
    graph.add_node("thought", thought_node)

    # 添加边
    graph.add_edge(START, "query")
    graph.add_edge("query", "context_manager")
    graph.add_edge("context_manager", "rag")
    graph.add_edge("rag", "chat")
    graph.add_edge("chat", "thought")
    graph.add_edge("thought", END)

    # 编译，绑定 MemorySaver 实现短期记忆
    app = graph.compile(checkpointer=memory)
    return app
