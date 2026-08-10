"""
Agent 节点实现 - 带上下文管理
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agent.state import AgentState
from agent.prompts import (
    REPHRASE_PROMPT,
    CHAT_PROMPT,
    MINDMAP_PROMPT,
)
from agent.context_manager import ContextManager, estimate_tokens
from agent.context_tracker import get_tracker
from rag.pipeline import RAGPipeline

load_dotenv()

# 全局 LLM 实例
_llm = None
_rag_pipeline = None
_context_manager = ContextManager()


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("BASE_URL"),
            model=os.getenv("LLM_MODEL"),
            temperature=0.1,
        )
    return _llm


def get_rag_pipeline(vs_manager, bm25_index):
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(
            llm=get_llm(),
            vs_manager=vs_manager,
            bm25_index=bm25_index,
            rephrase_prompt=REPHRASE_PROMPT,
            response_prompt=CHAT_PROMPT
        )
    return _rag_pipeline


def query_node(state: AgentState) -> AgentState:
    """接收并初始化输入，追加到消息历史"""
    user_input = state.get("user_input", "")
    session_id = state.get("session_id", "")

    # 如果没有 user_input，尝试从 messages 中提取
    if not user_input:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        if last_message and hasattr(last_message, 'content'):
            user_input = last_message.content

    # 将当前输入追加到消息历史
    from langchain_core.messages import HumanMessage
    messages = list(state.get("messages", []))
    if user_input and (not messages or messages[-1].content != user_input):
        messages.append(HumanMessage(content=user_input))

    return {
        **state,
        "user_input": user_input,
        "session_id": session_id,
        "messages": messages,
        "retrieved_docs": [],
        "rephrased_query": "",
        "rag_relevance_scores": [],
        "rag_context_compressed": False,
        "rag_deduplicated": False,
        "context_summary": "",
        "context_token_count": 0,
        "context_truncated": False,
        "current_topics": [],
        "subject_history": [],
        "knowledge_gaps": [],
        "conversation_summary": "",
        "turn_count": 0,
        "answer": "",
        "mindmap_mermaid": "",
    }


def context_manager_node(state: AgentState) -> AgentState:
    """上下文管理节点：历史摘要 + Token 预算控制"""
    messages = list(state.get("messages", []))
    llm = get_llm()

    # 管理历史消息
    managed_messages, context_summary, token_count = _context_manager.manage_history(
        messages, llm
    )

    return {
        **state,
        "messages": managed_messages,
        "context_summary": context_summary,
        "context_token_count": token_count,
    }


def rag_node(state: AgentState) -> AgentState:
    """检索 RAG 知识库（增强版：含相关性评分和过滤）"""
    query = state.get("user_input", "")
    current_topics = state.get("current_topics", [])

    # 构建 RAG pipeline
    from rag.vectorstore import VectorStoreManager
    from rag.bm25_index import BM25Index

    vs_manager = VectorStoreManager()
    bm25_index = BM25Index()

    if not bm25_index.load_cache():
        data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        if os.path.exists(data_folder):
            from rag.ingestor import KBIngestor
            ingestor = KBIngestor(data_folder, vs_manager, bm25_index)
            ingestor.sync()

    pipeline = get_rag_pipeline(vs_manager, bm25_index)

    # 检索（原始结果）
    retrieved_docs = pipeline.retrieve(query)

    # 增强：相关性评分和过滤
    scored_docs, scores = _score_and_filter_docs(retrieved_docs, query)

    # 增强：压缩长文档
    compressed_docs = _compress_docs(scored_docs)

    # 使用学科背景增强 query 重写（仅在有上下文时）
    subject_hint = ", ".join(current_topics[-3:]) if current_topics else ""
    rephrased = query
    if subject_hint:
        from agent.prompts import REPHRASE_PROMPT
        enhanced_prompt = REPHRASE_PROMPT.partial(subject_hint=subject_hint)
        try:
            chain = enhanced_prompt | get_llm()
            rephrased = chain.invoke({"question": query}).content.strip()
        except Exception:
            rephrased = query

    return {
        **state,
        "retrieved_docs": compressed_docs,
        "rag_relevance_scores": scores,
        "rag_context_compressed": True,
        "rag_deduplicated": len(retrieved_docs) != len(compressed_docs),
        "rephrased_query": rephrased,
    }


def _score_and_filter_docs(docs: list, query: str, threshold: float = 2.0) -> tuple:
    """
    对检索到的文档进行相关性评分和过滤

    Returns:
        (filtered_docs, scores)
    """
    if not docs:
        return [], []

    llm = get_llm()
    from agent.prompts import RELEVANCE_PROMPT

    scored = []
    for doc in docs:
        try:
            chain = RELEVANCE_PROMPT | llm
            result = chain.invoke({"query": query, "doc": doc})
            score_text = result.content.strip()
            # 提取分数
            try:
                score = float(score_text)
            except ValueError:
                score = 3  # 默认中等分数
        except Exception:
            score = 3  # LLM 调用失败时默认中等分数

        if score >= threshold:
            scored.append({"content": doc, "score": score})

    contents = [s["content"] for s in scored]
    scores = [s["score"] for s in scored]
    return contents, scores


def _compress_docs(docs: list, max_chars_per_doc: int = 400) -> list:
    """
    压缩文档：去除重复内容，截断超长文档
    """
    compressed = []
    seen_keys = set()

    for doc in docs:
        # 简单去重：检查前50个字符是否已出现过
        key = doc[:max_chars_per_doc // 2]
        if key not in seen_keys:
            seen_keys.add(key)
            # 截断超长文档
            if len(doc) > max_chars_per_doc:
                doc = doc[:max_chars_per_doc] + "..."
            compressed.append(doc)

    return compressed


def chat_node(state: AgentState) -> AgentState:
    """生成答疑回答（使用完整对话历史 + 上下文管理）"""
    llm = get_llm()

    messages = list(state.get("messages", []))
    retrieved_docs = state.get("retrieved_docs", [])
    current_topics = state.get("current_topics", [])
    knowledge_gaps = state.get("knowledge_gaps", [])
    conversation_summary = state.get("conversation_summary", "")

    # 构建上下文
    context = "\n\n".join(retrieved_docs) if retrieved_docs else "无额外上下文，请依靠模型自身知识解答。"

    # 使用 ContextManager 构建带预算的消息列表
    context_result = _context_manager.assemble_context(
        messages,
        rag_context=context,
        llm=llm,
    )
    final_messages = context_result["messages"]
    token_count = context_result["token_count"]
    is_truncated = context_result["truncated"]

    # 构建系统提示
    topics_str = ", ".join(current_topics) if current_topics else "无"
    gaps_str = ", ".join(knowledge_gaps) if knowledge_gaps else "无"
    summary_str = conversation_summary if conversation_summary else "新对话"

    system_msg = {
        "role": "system",
        "content": f"""你是一个友好的初中教育智能体，名叫"小智老师"。

你的职责：
1. 根据提供的参考资料解答学生的问题
2. 回答要通俗易懂，适合初中生理解
3. 如果参考资料中没有相关信息，诚实地告知学生
4. 鼓励性的语言，帮助学生建立学习信心
5. 记住之前的对话内容，进行多轮连贯对话
6. 关注学生可能存在的知识薄弱点，给予针对性指导

回答格式：
- 先给出清晰的结论
- 然后详细解释
- 如有必要，给出例题或记忆技巧

当前学习状态：
- 正在讨论的知识点：{topics_str}
- 学生可能有困难的地方：{gaps_str}
- 对话摘要：{summary_str}"""
    }

    rag_context_msg = {
        "role": "system",
        "content": f"""【参考资料】
{context}

如果参考资料与问题无关，请忽略参考资料，直接回答问题。"""
    }

    # 构建完整消息链
    full_messages = [system_msg, rag_context_msg] + [
        {"role": msg.type, "content": msg.content}
        for msg in final_messages if msg.type in ("human", "ai")
    ]

    response = llm.invoke(full_messages)

    # 更新跨轮次追踪
    tracker = get_tracker()
    if state.get("session_id"):
        tracker_result = tracker.update_from_query(
            state["session_id"],
            final_messages,
            llm
        )
        current_topics = tracker_result.get("current_topics", current_topics)
        knowledge_gaps = tracker_result.get("knowledge_gaps", knowledge_gaps)
        conversation_summary = tracker_result.get("conversation_summary", conversation_summary)

    return {
        **state,
        "answer": response.content,
        "context_token_count": token_count,
        "context_truncated": is_truncated,
        "current_topics": current_topics,
        "knowledge_gaps": knowledge_gaps,
        "conversation_summary": conversation_summary,
    }


def thought_node(state: AgentState) -> AgentState:
    """生成知识点思维导图"""
    llm = get_llm()

    user_input = state.get("user_input", "")
    current_topics = state.get("current_topics", [])
    knowledge_gaps = state.get("knowledge_gaps", [])

    topics_str = ", ".join(current_topics) if current_topics else ""
    gaps_str = ", ".join(knowledge_gaps) if knowledge_gaps else ""

    prompt = MINDMAP_PROMPT.format(
        knowledge_points=user_input,
        current_topics=topics_str,
        knowledge_gaps=gaps_str,
    )

    response = llm.invoke(prompt)
    mermaid_code = response.content

    # 清理可能的前后标记
    for prefix in ["```mermaid", "```", "mermaid"]:
        if mermaid_code.startswith(prefix):
            mermaid_code = mermaid_code[len(prefix):].lstrip()
        if mermaid_code.endswith("```"):
            mermaid_code = mermaid_code[:-3].rstrip()

    return {
        **state,
        "mindmap_mermaid": mermaid_code,
    }
