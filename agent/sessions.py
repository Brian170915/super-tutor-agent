"""
会话管理（无用户体系）
"""
import uuid
import time
from typing import Dict, List, Optional


# 内存会话存储
_sessions: Dict[str, dict] = {}


def create_session() -> str:
    """创建新会话，返回 session_id"""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "messages": [],
        "session_start": time.time(),
    }
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """获取会话信息"""
    return _sessions.get(session_id)


def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[dict] = None
):
    """添加消息到会话"""
    if session_id not in _sessions:
        create_session()

    _sessions[session_id]["messages"].append({
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "metadata": metadata or {},
    })


def get_messages(session_id: str) -> List[dict]:
    """获取会话历史消息"""
    session = _sessions.get(session_id)
    if session:
        return session["messages"]
    return []


def clear_session(session_id: str):
    """清空会话"""
    if session_id in _sessions:
        _sessions[session_id]["messages"] = []
