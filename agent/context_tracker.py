"""
跨轮次上下文追踪器 — 学科/知识点/困难点追踪
"""
import threading
from typing import List, Optional, Dict
from langchain_core.messages import BaseMessage

from agent.prompts import TOPIC_IDENTIFY_PROMPT, GAP_IDENTIFY_PROMPT, SUMMARY_UPDATE_PROMPT


class SessionTracker:
    """单个会话的追踪状态"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.subject_history: List[str] = []       # 历史学科列表
        self.current_topics: List[str] = []         # 当前讨论的知识点
        self.knowledge_gaps: List[str] = []         # 学生可能有困难的主题
        self.conversation_summary: str = ""         # 对话摘要
        self.turn_count: int = 0                    # 对话轮次
        self.last_query: str = ""                   # 上一次问题
        self.last_subject: Optional[str] = None     # 最后讨论的学科

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "subject_history": self.subject_history,
            "current_topics": self.current_topics,
            "knowledge_gaps": self.knowledge_gaps,
            "conversation_summary": self.conversation_summary,
            "turn_count": self.turn_count,
            "last_subject": self.last_subject,
        }


class CrossTurnTracker:
    """
    跨轮次上下文追踪器（线程安全）

    跟踪每个会话的：
    - 学科/知识点变化
    - 学生可能的困难点
    - 对话历史摘要
    """

    def __init__(self):
        self._trackers: Dict[str, SessionTracker] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionTracker:
        """获取或创建会话追踪器"""
        with self._lock:
            if session_id not in self._trackers:
                self._trackers[session_id] = SessionTracker(session_id)
            return self._trackers[session_id]

    def cleanup(self, session_id: str):
        """清理会话追踪器"""
        with self._lock:
            self._trackers.pop(session_id, None)

    def identify_topics(self, messages: List[BaseMessage], llm=None) -> dict:
        """
        从最新消息中提取当前学科和知识点

        Returns:
            {"subjects": [...], "topics": [...]}
        """
        if not llm:
            return {"subjects": [], "topics": []}

        # 取最近 6 条消息
        recent = [m for m in messages[-6:] if m.type in ("human", "ai")]
        if not recent:
            return {"subjects": [], "topics": []}

        history_text = "\n".join(
            f"{'学生' if m.type == 'human' else '老师'}: {m.content[:200]}"
            for m in recent
        )

        try:
            chain = TOPIC_IDENTIFY_PROMPT | llm
            result = chain.invoke({"messages": history_text})
            content = result.content.strip()

            # 简单解析输出
            subjects = []
            topics = []
            for line in content.split("\n"):
                line = line.strip()
                if "学科" in line or "科目" in line:
                    # 提取学科
                    parts = line.split("：")
                    if len(parts) > 1:
                        subjects = [s.strip() for s in parts[1].split(",")]
                elif "知识点" in line or "知识点" in line:
                    parts = line.split("：")
                    if len(parts) > 1:
                        topics = [t.strip() for t in parts[1].split(",")]

            return {"subjects": subjects, "topics": topics}
        except Exception:
            return {"subjects": [], "topics": []}

    def detect_gaps(self, messages: List[BaseMessage], llm=None) -> List[str]:
        """
        检测学生可能存在的知识薄弱点

        检测逻辑：
        1. 同一知识点反复提问
        2. 追问细节
        3. 表达困惑（"还是不懂"、"不太明白"等）
        """
        gaps = []

        # 规则1：检测困惑表达（中英文）
        confusion_keywords = ["不懂", "不明白", "不会", "困惑", "困难", "难", "还是不懂", "没懂",
                              "don't understand", "don't get it", "confused", "still not", "hard"]
        recent_human = [m.content for m in messages[-4:] if m.type == "human"]
        for content in recent_human:
            for kw in confusion_keywords:
                if kw in content:
                    gaps.append(f"困惑表达: {content[:50]}")
                    break

        # 规则2：检测重复提问（相同知识点多次出现）
        human_contents = [m.content for m in messages if m.type == "human"]
        seen_topics = {}
        for content in human_contents:
            # 简单关键词匹配
            for keyword in ["什么", "怎么", "为何", "如何", "为什么"]:
                if keyword in content:
                    if content[:30] in seen_topics:
                        seen_topics[content[:30]] += 1
                    else:
                        seen_topics[content[:30]] = 1

        repeated = [k for k, v in seen_topics.items() if v > 2]
        gaps.extend([f"重复提问: {r[:40]}" for r in repeated])

        # 规则3：LLM 辅助检测
        if llm:
            recent = [m for m in messages[-6:] if m.type in ("human", "ai")]
            if recent:
                history_text = "\n".join(
                    f"{'学生' if m.type == 'human' else '老师'}: {m.content[:200]}"
                    for m in recent
                )
                try:
                    chain = GAP_IDENTIFY_PROMPT | llm
                    result = chain.invoke({"messages": history_text})
                    content = result.content.strip()
                    if content and content != "无":
                        llm_gaps = [g.strip() for g in content.split(",") if g.strip()]
                        gaps.extend(llm_gaps)
                except Exception:
                    pass

        # 去重
        return list(set(gaps))[:5]  # 最多保留5个困难点

    def update_summary(self, messages: List[BaseMessage], llm=None):
        """增量更新对话摘要"""
        tracker = self._trackers.get("main")
        if not tracker:
            return

        tracker.turn_count += 1

        # 构建当前对话文本
        all_messages = [m for m in messages if m.type in ("human", "ai")]
        history_text = "\n".join(
            f"{'学生' if m.type == 'human' else '老师'}: {m.content}"
            for m in all_messages[-10:]
        )

        if llm and SUMMARY_UPDATE_PROMPT:
            try:
                chain = SUMMARY_UPDATE_PROMPT | llm
                result = chain.invoke({
                    "summary": tracker.conversation_summary,
                    "new_messages": history_text,
                })
                tracker.conversation_summary = result.content.strip()
            except Exception:
                pass

    def update_from_query(self, session_id: str, messages: List[BaseMessage], llm=None):
        """
        基于新消息更新会话追踪状态

        调用顺序：
        1. 识别学科和知识点
        2. 检测困难点
        3. 更新摘要
        """
        tracker = self.get_or_create(session_id)

        # 识别主题
        topic_result = self.identify_topics(messages, llm)
        if topic_result["subjects"]:
            tracker.last_subject = topic_result["subjects"][0]
            for subj in topic_result["subjects"]:
                if subj not in tracker.subject_history:
                    tracker.subject_history.append(subj)

        if topic_result["topics"]:
            # 更新当前知识点（去重，保留最新）
            for topic in topic_result["topics"]:
                if topic not in tracker.current_topics:
                    tracker.current_topics.append(topic)
                else:
                    tracker.current_topics.remove(topic)
                    tracker.current_topics.append(topic)
            # 限制当前知识点数量
            tracker.current_topics = tracker.current_topics[-5:]

        # 检测困难点
        gaps = self.detect_gaps(messages, llm)
        for gap in gaps:
            if gap not in tracker.knowledge_gaps:
                tracker.knowledge_gaps.append(gap)
        tracker.knowledge_gaps = tracker.knowledge_gaps[-3:]  # 最多3个

        # 更新摘要
        self.update_summary(messages, llm)

        # 同步到 AgentState
        return {
            "current_topics": tracker.current_topics,
            "subject_history": tracker.subject_history,
            "knowledge_gaps": tracker.knowledge_gaps,
            "conversation_summary": tracker.conversation_summary,
            "turn_count": tracker.turn_count,
        }


# 全局单例
_tracker_instance = None
_tracker_lock = threading.Lock()


def get_tracker() -> CrossTurnTracker:
    """获取全局追踪器单例"""
    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = CrossTurnTracker()
        return _tracker_instance
