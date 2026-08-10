"""
LangGraph 状态图构建 - 带意图识别和动态路由
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState
from agent.nodes import (
    query_node, intent_node, context_manager_node, rag_node,
    chat_node, thought_node, route_by_intent, route_after_chat
)

# 短期记忆：每个 session_id 对应独立的历史记录
memory = MemorySaver()


def build_graph():
    """构建并编译 Agent 状态图（带意图路由）"""
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("query", query_node)
    graph.add_node("intent", intent_node)
    graph.add_node("context_manager", context_manager_node)
    graph.add_node("rag", rag_node)
    graph.add_node("chat", chat_node)
    graph.add_node("thought", thought_node)

    # 添加边
    graph.add_edge(START, "query")
    graph.add_edge("query", "intent")

    # 根据意图条件路由
    graph.add_conditional_edges(
        "intent",
        route_by_intent,
        {"rag": "rag", "chat": "chat"}
    )

    # RAG 路径：rag → context_manager → chat
    graph.add_edge("rag", "context_manager")
    graph.add_edge("context_manager", "chat")

    # chat 之后根据意图决定是否生成思维导图
    graph.add_conditional_edges(
        "chat",
        route_after_chat,
        {"thought": "thought", END: END}
    )

    # 思维导图 → END
    graph.add_edge("thought", END)

    # 编译，绑定 MemorySaver 实现短期记忆
    app = graph.compile(checkpointer=memory)
    return app
