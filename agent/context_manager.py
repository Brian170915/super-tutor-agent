"""
上下文管理器 — Token 感知、历史摘要、滑动窗口
"""
import tiktoken
from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from agent.prompts import SUMMARIZE_PROMPT

# Token 预算配置
CONTEXT_CONFIG = {
    "max_total_tokens": 8000,          # 总 Token 预算
    "system_prompt_tokens": 600,       # System prompt 预留
    "rag_context_tokens": 2000,        # RAG 上下文预留
    "summary_max_tokens": 400,         # 历史摘要最大 Token 数
    "min_message_tokens": 300,         # 每条消息最小 Token 数
}

# 对话历史管理阈值
SUMMARIZE_THRESHOLD = 12   # 消息数超过此值触发摘要
WINDOW_SIZE = 10           # 保留最近 N 条原始消息


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """估算文本的 Token 数量"""
    if not text:
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def estimate_tokens_messages(messages: List[BaseMessage]) -> int:
    """估算消息列表的 Token 数量"""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content)
    return total


class ContextManager:
    """上下文管理器：历史摘要 + 滑动窗口 + Token 预算控制"""

    def __init__(self, config: Optional[dict] = None):
        self.config = {**CONTEXT_CONFIG, **(config or {})}

    def manage_history(self, messages: List[BaseMessage], llm=None) -> tuple:
        """
        管理对话历史：
        - 消息数 <= WINDOW_SIZE: 保留全部原始消息
        - WINDOW_SIZE < 消息数 <= SUMMARIZE_THRESHOLD: 滑动窗口，保留最近 WINDOW_SIZE 条
        - 消息数 > SUMMARIZE_THRESHOLD: 滑动窗口 + 摘要旧消息

        Returns:
            (managed_messages, context_summary, token_count)
        """
        human_ai_messages = [m for m in messages if m.type in ("human", "ai")]
        total_count = len(human_ai_messages)

        if total_count <= WINDOW_SIZE:
            # 消息很少，全部保留
            return list(messages), "", estimate_tokens_messages(messages)

        # 保留最近 WINDOW_SIZE 条原始消息
        window_messages = human_ai_messages[-WINDOW_SIZE:]

        # 需要摘要的旧消息
        older_messages = human_ai_messages[:-WINDOW_SIZE]

        if total_count <= SUMMARIZE_THRESHOLD:
            # 不超过阈值，直接滑动窗口，不需要摘要
            context_summary = ""
            summarized_msg = None
        else:
            # 需要摘要：将旧消息压缩
            context_summary = self._summarize_history(older_messages, llm)
            summarized_msg = SystemMessage(
                content=f"【历史对话摘要】\n{context_summary}\n\n以下是最近的对话内容："
            )

        # 构建最终消息列表
        if summarized_msg:
            managed = [summarized_msg] + list(window_messages)
        else:
            managed = list(window_messages)

        token_count = estimate_tokens_messages(managed)
        return managed, context_summary, token_count

    def _summarize_history(self, messages: List[BaseMessage], llm=None) -> str:
        """使用 LLM 将旧消息压缩为摘要"""
        if not llm:
            # 无 LLM 时做简单截断
            return self._truncate_summary(messages)

        # 构建历史文本
        history_text = "\n".join(
            f"{'学生' if m.type == 'human' else '老师'}: {m.content}"
            for m in messages
        )

        chain = SUMMARIZE_PROMPT | llm
        result = chain.invoke({"history": history_text})
        summary = result.content.strip()

        # 确保不超过摘要 Token 预算
        if estimate_tokens(summary) > self.config["summary_max_tokens"]:
            summary = self._truncate_summary(messages)

        return summary

    def _truncate_summary(self, messages: List[BaseMessage]) -> str:
        """无 LLM 时的降级摘要：提取关键信息"""
        lines = []
        for m in messages[-6:]:  # 只取最近6条
            role = "学生" if m.type == "human" else "老师"
            content = m.content[:100]  # 每条最多100字
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def assemble_context(
        self,
        messages: List[BaseMessage],
        rag_context: str = "",
        rag_scores: List[float] = None,
        llm=None,
    ) -> dict:
        """
        构建带 Token 预算的完整上下文

        Returns:
            {
                "messages": [...],           # 最终消息列表
                "token_count": int,          # 总 Token 数
                "truncated": bool,           # 是否被截断
                "context_summary": str,      # 历史摘要
            }
        """
        budget = self.config["max_total_tokens"]
        reserved = self.config["system_prompt_tokens"] + self.config["rag_context_tokens"]
        available_for_messages = budget - reserved - estimate_tokens(rag_context)

        # 管理历史消息
        managed_messages, summary, msg_token_count = self.manage_history(messages, llm)

        truncated = False

        # 如果消息 Token 超出预算，逐步截断
        if msg_token_count > available_for_messages:
            # 先尝试减少窗口
            while len(managed_messages) > 2 and msg_token_count > available_for_messages:
                managed_messages = managed_messages[1:]
                msg_token_count = estimate_tokens_messages(managed_messages)
            truncated = True

        return {
            "messages": managed_messages,
            "token_count": estimate_tokens_messages(managed_messages) + estimate_tokens(rag_context),
            "truncated": truncated,
            "context_summary": summary,
        }
