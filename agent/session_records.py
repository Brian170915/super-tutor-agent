"""
练习/测试记录存储模块
使用 SQLite 持久化 quiz/test 结果
数据库路径: learning.db（项目根目录）
"""
import os
import re
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "learning.db")


class SessionRecordStore:
    """练习记录存储（线程安全，SQLite）"""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """建表（幂等）"""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quiz_records (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    questions   TEXT NOT NULL,
                    answers     TEXT NOT NULL,
                    correct_count INTEGER NOT NULL,
                    total_count INTEGER NOT NULL,
                    timestamp   TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quiz_session
                ON quiz_records (session_id)
            """)
            conn.commit()
        finally:
            conn.close()

    def save_quiz_result(
        self,
        session_id: str,
        questions: List[Dict[str, Any]],
        answers: Dict[int, str],
        correct_count: int,
    ) -> dict:
        """保存练习结果"""
        total = len(questions)
        record = {
            "session_id": session_id,
            "questions": questions,
            "answers": answers,
            "correct_count": correct_count,
            "total_count": total,
            "accuracy": correct_count / total if total > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO quiz_records
                        (session_id, questions, answers, correct_count, total_count, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        str(questions),
                        str(answers),
                        correct_count,
                        len(questions),
                        record["timestamp"],
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return record

    def get_session_records(self, session_id: str) -> List[Dict[str, Any]]:
        """获取指定会话的练习记录"""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM quiz_records WHERE session_id = ? ORDER BY timestamp",
                    (session_id,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_all_session_summaries(self, tracker) -> List[Dict[str, Any]]:
        """获取所有会话的练习统计摘要（跨会话聚合用）"""
        result = []
        # 从 tracker 获取已有会话
        sids_from_tracker = set(tracker._trackers.keys())
        # 从数据库获取所有有练习记录的会话
        sids_from_db = set(self.get_all_session_ids())
        all_sids = sids_from_tracker | sids_from_db

        for sid in all_sids:
            summary = self.get_session_summary(sid)
            st = tracker._trackers.get(sid)
            subject_history = []
            turn_count = 0
            last_quiz_score = None
            session_start_time = ""
            if st:
                subject_history = st.subject_history or self.infer_subjects_from_records(sid)
                turn_count = st.turn_count
                last_quiz_score = st.last_quiz_score
                session_start_time = st.session_start_time
            else:
                # 从数据库记录推断
                subject_history = self.infer_subjects_from_records(sid)
                # 从最新记录的 timestamp 推断会话开始时间
                records = self.get_session_records(sid)
                if records:
                    session_start_time = records[0]["timestamp"]
            result.append({
                "session_id": sid,
                "subject_history": subject_history,
                "turn_count": turn_count,
                "last_quiz_score": last_quiz_score,
                "quiz_summary": summary,
                "session_start_time": session_start_time,
            })
        return result

    def get_session_summaries(self, session_ids: List[str]) -> List[Dict[str, Any]]:
        """获取指定会话列表的练习统计摘要"""
        result = []
        for sid in session_ids:
            summary = self.get_session_summary(sid)
            result.append({
                "session_id": sid,
                "quiz_summary": summary,
            })
        return result

    def get_all_session_ids(self) -> List[str]:
        """获取所有有练习记录的 session_id"""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT DISTINCT session_id FROM quiz_records ORDER BY timestamp DESC"
                ).fetchall()
                return [r["session_id"] for r in rows]
            finally:
                conn.close()

    def get_all_records(self) -> List[Dict[str, Any]]:
        """获取所有记录"""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM quiz_records ORDER BY timestamp"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

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

    def infer_subjects_from_records(self, session_id: str) -> List[str]:
        """从练习记录的题目文本中推断学科"""
        records = self.get_session_records(session_id)
        if not records:
            return []

        # 关键词 → 学科映射
        subject_keywords = {
            "数学": ["函数", "方程", "不等式", "几何", "三角形", "圆", "坐标", "概率", "统计",
                     "有理数", "整式", "分式", "勾股", "相似", "全等", "平行四边形", "二次",
                     "指数", "对数", "三角函数", "导数", "积分", "向量", "数列"],
            "物理": ["力", "速度", "加速度", "质量", "密度", "压强", "浮力", "摩擦力",
                     "重力", "弹力", "运动", "能量", "功", "功率", "电", "磁", "光",
                     "声", "透镜", "电路", "欧姆", "电阻", "电流", "电压", "功率",
                     "牛顿", "惯性", " momentum", "动量", "冲量", "机械波"],
            "化学": ["化学", "分子", "原子", "元素", "化合", "反应", "氧化", "还原",
                     "酸", "碱", "盐", "离子", "溶液", "浓度", "pH", "金属", "非金属",
                     "碳", "氧", "氢", "氮", "方程式", "化学变化"],
            "语文": ["古诗", "古文", "文言文", "诗词", "阅读", "作文", "修辞", "修辞手法",
                     "阅读理解", "名著", "诗歌", "散文", "议论文", "说明文", "修辞", "成语"],
            "英语": ["English", "grammar", "tense", "verb", "noun", "sentence", "reading",
                     "writing", "vocabulary", "dialogue", "translation", "past tense",
                     "present tense", "past participle", "comparative", "superlative"],
        }

        # 收集所有题目文本
        all_text = ""
        for r in records:
            try:
                questions = eval(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
                for q in questions:
                    if isinstance(q, dict):
                        all_text += q.get("question", "") + " "
                        for opt in q.get("options", []):
                            all_text += opt + " "
                        if q.get("explanation"):
                            all_text += q["explanation"] + " "
            except Exception:
                pass

        # 统计各学科关键词出现次数
        scores = {}
        for subject, keywords in subject_keywords.items():
            count = 0
            for kw in keywords:
                count += len(re.findall(re.escape(kw), all_text, re.IGNORECASE))
            if count > 0:
                scores[subject] = count

        # 返回得分最高的学科（最多3个）
        sorted_subjects = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_subjects[:3]]


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
