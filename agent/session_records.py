"""
练习/测试记录存储模块
使用 JSON 文件持久化 quiz/test 结果
"""
import json
import os
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any


RECORDS_FILE = os.path.join("data", "session_records.json")


class SessionQuizRecord:
    """单次练习记录"""

    def __init__(
        self,
        session_id: str,
        questions: List[Dict[str, Any]],
        answers: Dict[int, str],
        correct_count: int,
        total_count: int,
        timestamp: str,
    ):
        self.session_id = session_id
        self.questions = questions
        self.answers = answers
        self.correct_count = correct_count
        self.total_count = total_count
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "questions": self.questions,
            "answers": self.answers,
            "correct_count": self.correct_count,
            "total_count": self.total_count,
            "accuracy": self.correct_count / self.total_count if self.total_count > 0 else 0,
            "timestamp": self.timestamp,
        }


class SessionRecordStore:
    """练习记录存储（线程安全）"""

    def __init__(self, filepath: str = RECORDS_FILE):
        self._filepath = filepath
        self._lock = threading.Lock()
        self._records: List[SessionQuizRecord] = []
        self._load()

    def _load(self):
        """从文件加载记录"""
        if os.path.exists(self._filepath):
            try:
                with open(self._filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    self._records.append(SessionQuizRecord(**item))
            except Exception:
                self._records = []

    def _save(self):
        """保存到文件"""
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self._records], f, ensure_ascii=False, indent=2)

    def save_quiz_result(
        self,
        session_id: str,
        questions: List[Dict[str, Any]],
        answers: Dict[int, str],
        correct_count: int,
    ):
        """保存练习结果"""
        record = SessionQuizRecord(
            session_id=session_id,
            questions=questions,
            answers=answers,
            correct_count=correct_count,
            total_count=len(questions),
            timestamp=datetime.now().isoformat(),
        )
        with self._lock:
            self._records.append(record)
            self._save()
        return record

    def get_session_records(self, session_id: str) -> List[Dict[str, Any]]:
        """获取指定会话的练习记录"""
        with self._lock:
            return [r.to_dict() for r in self._records if r.session_id == session_id]

    def get_all_records(self) -> List[Dict[str, Any]]:
        """获取所有记录"""
        with self._lock:
            return [r.to_dict() for r in self._records]

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """获取会话的练习统计摘要"""
        records = self.get_session_records(session_id)
        if not records:
            return {"total_quizzes": 0, "total_questions": 0, "total_correct": 0, "avg_accuracy": 0}
        total_q = sum(r["total_count"] for r in records)
        total_c = sum(r["correct_count"] for r in records)
        return {
            "total_quizzes": len(records),
            "total_questions": total_q,
            "total_correct": total_c,
            "avg_accuracy": total_c / total_q if total_q > 0 else 0,
            "records": records,
        }


# 全局单例
_store_instance = None
_store_lock = threading.Lock()


def get_record_store() -> SessionRecordStore:
    """获取记录存储单例"""
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            _store_instance = SessionRecordStore()
        return _store_instance
