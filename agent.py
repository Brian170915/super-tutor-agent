"""
超级教育智能体 - 主入口（带上下文管理 + 功能面板）
"""
import os
import time
import json
import base64
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langserve import add_routes

from agent.graph import build_graph
from rag.knowledge_structure import KNOWLEDGE_STRUCTURE, get_subjects, get_grades, get_topics
from agent.context_tracker import get_tracker
from agent.session_records import get_record_store
from agent.prompts import QUIZ_BATCH_PROMPT, REPORT_GENERATION_PROMPT

load_dotenv()


class ChatRequest(BaseModel):
    session_id: str = ""
    user_input: str
    image_base64: str = ""


class QuizRequest(BaseModel):
    session_id: str = ""
    count: int = 5


class SubmitQuizRequest(BaseModel):
    session_id: str = ""
    questions: list = []
    answers: dict = {}
    correct_count: int = 0


app = FastAPI(
    title="超级教育智能体",
    description="基于 LangGraph 的初中教育智能体 - 答疑 + RAG + 思维导图 + 上下文管理",
    version="3.0.0"
)

# 编译 LangGraph（内部绑定 MemorySaver 管理对话历史）
graph = build_graph()

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 会话活动追踪：session_id -> 最后活跃时间戳
# 用于 MemorySaver 的 TTL 清理
_session_last_active: dict[str, float] = {}
SESSION_TTL_SECONDS = 1800  # 30 分钟未活动则自动清理


@app.on_event("startup")
async def startup():
    """启动时初始化 RAG 基础设施"""
    from rag.vectorstore import VectorStoreManager
    from rag.bm25_index import BM25Index

    vs_manager = VectorStoreManager()
    bm25_index = BM25Index()

    if not bm25_index.load_cache():
        data_folder = os.path.join(os.path.dirname(__file__), "data")
        if os.path.exists(data_folder):
            from rag.ingestor import KBIngestor
            ingestor = KBIngestor(data_folder, vs_manager, bm25_index)
            ingestor.sync()
            print(f"知识库初始化完成")
        else:
            print("警告：data 目录不存在，RAG 检索可能返回空结果")
    else:
        print("BM25 缓存加载成功")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """主对话端点 - RAG 检索 + 答疑 + 思维导图 + 上下文管理"""
    session_id = request.session_id or str(__import__("uuid").uuid4())
    user_input = request.user_input

    # 如果有图片 base64，先 OCR 识别
    if request.image_base64:
        try:
            from ocr.agnes_client import get_ocr_client
            ocr_client = get_ocr_client()
            image_bytes = base64.b64decode(request.image_base64)
            ocr_text = ocr_client.recognize(image_bytes)
            if ocr_text:
                user_input = f"[图片识别内容]\n{ocr_text}\n\n请帮我讲解以上内容。"
        except Exception as e:
            print(f"OCR 识别失败: {e}")

    # 更新活动追踪
    _session_last_active[session_id] = time.time()

    # 清理过期会话（MemorySaver + 活动记录）
    _cleanup_expired_sessions()

    # 运行 graph，thread_id=session_id 让 MemorySaver 维护对话历史
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(
        {"user_input": user_input, "session_id": session_id, "messages": []},
        config=config,
    )

    answer = result.get("answer", "")
    mindmap = result.get("mindmap_mermaid", "")

    # 将 graph 结果同步到 CrossTurnTracker，确保学习报告和导出笔记能读取最新数据
    tracker = get_tracker()
    st = tracker.get_or_create(session_id)
    st.turn_count = result.get("turn_count", st.turn_count)
    st.current_topics = result.get("current_topics", st.current_topics)
    st.knowledge_gaps = result.get("knowledge_gaps", st.knowledge_gaps)
    st.subject_history = result.get("subject_history", st.subject_history)
    st.conversation_summary = result.get("conversation_summary", st.conversation_summary)

    return {
        "session_id": session_id,
        "answer": answer,
        "mindmap_mermaid": mindmap,
        "context": {
            "current_topics": result.get("current_topics", []),
            "knowledge_gaps": result.get("knowledge_gaps", []),
            "turn_count": result.get("turn_count", 0),
            "context_summary": result.get("context_summary", ""),
            "context_token_count": result.get("context_token_count", 0),
            "context_truncated": result.get("context_truncated", False),
        }
    }


def _cleanup_expired_sessions():
    """清理超过 TTL 未活动的会话，释放 MemorySaver 内存"""
    now = time.time()
    expired = [
        sid for sid, last in _session_last_active.items()
        if now - last > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        try:
            graph.get_checkpointer().delete_thread(sid)
        except Exception:
            pass
        del _session_last_active[sid]
        # 同时清理跨轮次追踪器
        tracker = get_tracker()
        tracker.cleanup(sid)


@app.get("/knowledge-structure")
async def knowledge_structure_endpoint(subject: str = None, grade: str = None):
    """获取知识点体系"""
    if subject:
        if grade:
            topics = get_topics(subject, grade)
            return {"subject": subject, "grade": grade, "topics": topics}
        else:
            grades = get_grades(subject)
            return {"subject": subject, "grades": grades}
    else:
        subjects = get_subjects()
        return {"subjects": subjects}


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    """上传知识文档到 RAG 知识库"""
    from rag.vectorstore import VectorStoreManager
    from rag.bm25_index import BM25Index
    from rag.ingestor import KBIngestor

    data_folder = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_folder, exist_ok=True)

    file_path = os.path.join(data_folder, file.filename)
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)

    vs_manager = VectorStoreManager()
    bm25_index = BM25Index()
    ingestor = KBIngestor(data_folder, vs_manager, bm25_index)
    added = ingestor.sync()

    return {
        "filename": file.filename,
        "added_chunks": added,
        "status": "success"
    }


@app.get("/session/{session_id}")
async def session_endpoint(session_id: str):
    """获取会话历史 - 从 MemorySaver 读取完整对话"""
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        return {
            "session_id": session_id,
            "messages": [
                {"role": msg.type, "content": msg.content}
                for msg in messages if msg.type in ("human", "ai")
            ],
            "context_summary": state.values.get("context_summary", ""),
            "current_topics": state.values.get("current_topics", []),
            "subject_history": state.values.get("subject_history", []),
            "knowledge_gaps": state.values.get("knowledge_gaps", []),
            "turn_count": state.values.get("turn_count", 0),
        }
    except Exception:
        return {"session_id": session_id, "messages": []}


@app.get("/context/{session_id}")
async def context_endpoint(session_id: str):
    """获取会话上下文信息（调试用）"""
    tracker = get_tracker()
    session_tracker = tracker.get_or_create(session_id)

    # 也尝试从 LangGraph 获取
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        langgraph_context = state.values
    except Exception:
        langgraph_context = {}

    return {
        "session_id": session_id,
        "tracker": session_tracker.to_dict(),
        "langgraph_state": {
            "context_summary": langgraph_context.get("context_summary", ""),
            "context_token_count": langgraph_context.get("context_token_count", 0),
            "context_truncated": langgraph_context.get("context_truncated", False),
            "rag_relevance_scores": langgraph_context.get("rag_relevance_scores", []),
            "turn_count": langgraph_context.get("turn_count", 0),
        }
    }


@app.post("/upload-image")
async def upload_image_endpoint(
    file: UploadFile = File(...),
    session_id: str = ""
):
    """上传图片并 OCR 识别"""
    if not session_id:
        session_id = str(__import__("uuid").uuid4())
    _session_last_active[session_id] = time.time()

    # 读取图片
    content = await file.read()

    # OCR 识别
    try:
        from ocr.agnes_client import get_ocr_client
        ocr_client = get_ocr_client()
        recognized_text = ocr_client.recognize(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 识别失败: {str(e)}")

    # 图片转 base64 预览
    image_preview = base64.b64encode(content).decode('utf-8')

    return {
        "session_id": session_id,
        "recognized_text": recognized_text,
        "image_preview": f"data:{file.content_type};base64,{image_preview}"
    }


@app.post("/generate-quiz")
async def generate_quiz_endpoint(request: QuizRequest):
    """基于当前知识点批量出题"""
    session_id = request.session_id or str(__import__("uuid").uuid4())
    _session_last_active[session_id] = time.time()

    # 从 tracker 获取学习上下文
    tracker = get_tracker()
    st = tracker.get_or_create(session_id)

    # 也尝试从 LangGraph 获取最新状态
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        st.current_topics = state.values.get("current_topics", st.current_topics)
        st.knowledge_gaps = state.values.get("knowledge_gaps", st.knowledge_gaps)
        st.subject_history = state.values.get("subject_history", st.subject_history)
    except Exception:
        pass

    topics = ", ".join(st.current_topics) if st.current_topics else "初中各科知识点"
    gaps = ", ".join(st.knowledge_gaps) if st.knowledge_gaps else "无"
    subjects = ", ".join(st.subject_history) if st.subject_history else "初中各科"

    # 获取会话历史
    history_texts = []
    try:
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        for msg in messages[-10:]:
            if msg.type in ("human", "ai"):
                history_texts.append(f"{'学生' if msg.type == 'human' else '老师'}: {msg.content[:200]}")
    except Exception:
        pass

    history = "\n".join(history_texts) if history_texts else "无历史对话"

    # 调用 LLM 生成试卷
    from agent.nodes import get_llm
    llm = get_llm()
    chain = QUIZ_BATCH_PROMPT | llm
    result = chain.invoke({
        "topics": topics,
        "gaps": gaps,
        "subjects": subjects,
        "history": history,
        "count": request.count,
    })

    # 解析 JSON 响应
    content = result.content.strip()
    try:
        # 清理可能的 markdown 代码块
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3].rstrip()
        quiz_data = json.loads(content)
        questions = quiz_data.get("questions", [])
    except json.JSONDecodeError:
        # 降级：返回纯文本
        questions = [{"id": 1, "question": content, "options": [], "correct_answer": "", "explanation": ""}]

    return {
        "session_id": session_id,
        "questions": questions,
        "topics": st.current_topics,
    }


@app.post("/submit-quiz")
async def submit_quiz_endpoint(request: SubmitQuizRequest):
    """提交答题结果并保存"""
    session_id = request.session_id or str(__import__("uuid").uuid4())
    _session_last_active[session_id] = time.time()

    record_store = get_record_store()
    record = record_store.save_quiz_result(
        session_id=session_id,
        questions=request.questions,
        answers=request.answers,
        correct_count=request.correct_count,
    )

    # 同步到跨轮次追踪器
    tracker = get_tracker()
    tracker.add_quiz_record(session_id, record)

    return {
        "session_id": session_id,
        "total_count": record["total_count"],
        "correct_count": record["correct_count"],
        "accuracy": record["accuracy"],
    }


@app.get("/study-report/{session_id}")
async def study_report_endpoint(session_id: str):
    """获取学习报告数据"""
    tracker = get_tracker()
    st = tracker.get_or_create(session_id)

    # 也尝试从 LangGraph 获取最新状态
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        st.current_topics = state.values.get("current_topics", st.current_topics)
        st.knowledge_gaps = state.values.get("knowledge_gaps", st.knowledge_gaps)
        st.subject_history = state.values.get("subject_history", st.subject_history)
        st.turn_count = state.values.get("turn_count", st.turn_count)
        st.conversation_summary = state.values.get("context_summary", st.conversation_summary)
    except Exception:
        pass

    # 检查是否有数据
    has_data = st.turn_count > 0 or st.current_topics or st.subject_history

    if not has_data:
        return {
            "empty": True,
            "message": "多提问，报告会更丰富哦",
            "turn_count": 0,
            "current_topics": [],
            "knowledge_gaps": [],
            "subject_history": [],
            "conversation_summary": "",
        }

    return {
        "empty": False,
        "turn_count": st.turn_count,
        "current_topics": st.current_topics,
        "knowledge_gaps": st.knowledge_gaps,
        "subject_history": st.subject_history,
        "conversation_summary": st.conversation_summary,
    }


# ============ 学情分析报告相关端点 ============

@app.get("/api/report-data/{session_id}")
async def report_data_endpoint(session_id: str):
    """获取完整的学情分析报告数据"""
    tracker = get_tracker()
    st = tracker.get_or_create(session_id)
    record_store = get_record_store()

    # 从 LangGraph 获取最新状态
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        st.current_topics = state.values.get("current_topics", st.current_topics)
        st.knowledge_gaps = state.values.get("knowledge_gaps", st.knowledge_gaps)
        st.subject_history = state.values.get("subject_history", st.subject_history)
        st.turn_count = state.values.get("turn_count", st.turn_count)
        st.conversation_summary = state.values.get("context_summary", st.conversation_summary)
    except Exception:
        pass

    # 如果 tracker 中 subject_history 为空，从练习记录中推断学科
    if not st.subject_history:
        inferred = record_store.infer_subjects_from_records(session_id)
        if inferred:
            st.subject_history = inferred

    has_data = st.turn_count > 0 or st.current_topics or st.subject_history
    if not has_data:
        return {
            "empty": True,
            "message": "暂无学习数据，多提问来生成报告吧",
            "turn_count": 0,
            "current_topics": [],
            "knowledge_gaps": [],
            "subject_history": [],
            "conversation_summary": "",
            "mastery_levels": {},
            "quiz_summary": {"total_quizzes": 0, "total_questions": 0, "total_correct": 0, "avg_accuracy": 0},
            "improvement_suggestions": [],
            "strengths": [],
            "weaknesses": [],
            "next_steps": "",
            "grade_info": {},
        }

    # 获取练习记录
    quiz_summary = record_store.get_session_summary(session_id)

    # 计算知识点掌握度
    mastery_levels = _calculate_mastery(st, quiz_summary)

    # 获取对话历史（用于生成建议）
    try:
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        history_texts = [
            f"{'学生' if m.type == 'human' else '老师'}: {m.content[:150]}"
            for m in messages[-10:] if m.type in ("human", "ai")
        ]
        history_text = "\n".join(history_texts)
    except Exception:
        history_text = ""

    # 调用 LLM 生成改进建议
    improvement_suggestions = []
    strengths = []
    weaknesses = []
    next_steps = ""
    try:
        from agent.nodes import get_llm
        llm = get_llm()
        accuracy_pct = int(quiz_summary.get("avg_accuracy", 0) * 100)
        chain = REPORT_GENERATION_PROMPT | llm
        result = chain.invoke({
            "subjects": ", ".join(st.subject_history) if st.subject_history else "初中各科",
            "topics": ", ".join(st.current_topics) if st.current_topics else "无",
            "gaps": ", ".join(st.knowledge_gaps) if st.knowledge_gaps else "无",
            "summary": st.conversation_summary or "无摘要",
            "accuracy": accuracy_pct,
            "turns": st.turn_count,
        })
        content = result.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3].rstrip()
        report_data = json.loads(content)
        improvement_suggestions = report_data.get("improvement_suggestions", [])
        strengths = report_data.get("strengths", [])
        weaknesses = report_data.get("weaknesses", [])
        next_steps = report_data.get("next_steps", "")
    except Exception as e:
        print(f"生成改进建议失败: {e}")

    return {
        "empty": False,
        "turn_count": st.turn_count,
        "current_topics": st.current_topics,
        "knowledge_gaps": st.knowledge_gaps,
        "subject_history": st.subject_history,
        "conversation_summary": st.conversation_summary,
        "mastery_levels": mastery_levels,
        "quiz_summary": quiz_summary,
        "improvement_suggestions": improvement_suggestions,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "next_steps": next_steps,
        "session_start_time": st.session_start_time,
        "grade_info": _build_grade_info(st.current_topics),
    }


def _calculate_mastery(st, quiz_summary: dict) -> dict:
    """计算各知识点的掌握度分数"""
    mastery = {}
    base_score = 50

    # 基于对话轮次给基础分
    for topic in st.current_topics:
        mastery[topic] = base_score + min(st.turn_count * 3, 20)

    # 基于薄弱点扣分
    for gap in st.knowledge_gaps:
        for topic in st.current_topics:
            if topic in gap:
                mastery[topic] = max(0, mastery.get(topic, base_score) - 15)

    # 基于练习正确率调整
    avg_accuracy = quiz_summary.get("avg_accuracy", 0)
    for topic in mastery:
        mastery[topic] = min(100, max(0, mastery[topic] + int(avg_accuracy * 30 - 15)))

    return mastery


@app.get("/api/sessions")
async def list_sessions_endpoint():
    """获取所有会话列表"""
    tracker = get_tracker()
    record_store = get_record_store()
    sessions = []

    # 从 tracker 获取已有会话
    for sid, st in tracker._trackers.items():
        if not st.subject_history:
            inferred = record_store.infer_subjects_from_records(sid)
            if inferred:
                st.subject_history = inferred
        quiz_summary = record_store.get_session_summary(sid)
        sessions.append({
            "session_id": sid,
            "subject_history": st.subject_history,
            "current_topics": st.current_topics,
            "turn_count": st.turn_count,
            "conversation_summary": st.conversation_summary[:100] if st.conversation_summary else "",
            "session_start_time": st.session_start_time,
            "last_quiz_score": st.last_quiz_score,
            "quiz_summary": quiz_summary,
        })

    # 补充数据库中存在的、但 tracker 中已超时的会话
    db_sids = set(record_store.get_all_session_ids())
    tracker_sids = set(tracker._trackers.keys())
    for sid in db_sids - tracker_sids:
        quiz_summary = record_store.get_session_summary(sid)
        subject_history = record_store.infer_subjects_from_records(sid)
        records = record_store.get_session_records(sid)
        sessions.append({
            "session_id": sid,
            "subject_history": subject_history,
            "current_topics": [],
            "turn_count": 0,
            "conversation_summary": "",
            "session_start_time": records[0]["timestamp"] if records else "",
            "last_quiz_score": None,
            "quiz_summary": quiz_summary,
        })

    sessions.sort(key=lambda x: x.get("session_start_time", ""), reverse=True)
    return {"sessions": sessions}


@app.get("/api/sessions-summary")
async def sessions_summary_endpoint():
    """获取所有会话的聚合统计摘要"""
    tracker = get_tracker()
    record_store = get_record_store()

    # 优先从 tracker 获取，fallback 到数据库
    sessions = record_store.get_all_session_summaries(tracker)
    if not sessions:
        # 如果 tracker 全空，直接从数据库获取
        session_ids = record_store.get_all_session_ids()
        for sid in session_ids:
            quiz_summary = record_store.get_session_summary(sid)
            subject_history = record_store.infer_subjects_from_records(sid)
            records = record_store.get_session_records(sid)
            sessions.append({
                "session_id": sid,
                "subject_history": subject_history,
                "turn_count": 0,
                "last_quiz_score": None,
                "quiz_summary": quiz_summary,
                "session_start_time": records[0]["timestamp"] if records else "",
            })

    sessions.sort(key=lambda x: x.get("session_start_time", ""), reverse=True)

    # 按学科聚合
    subject_stats = {}
    for s in sessions:
        for subj in s["subject_history"]:
            if subj not in subject_stats:
                subject_stats[subj] = {"total_turns": 0, "total_quizzes": 0, "total_questions": 0, "total_correct": 0, "sessions": 0}
            subject_stats[subj]["total_turns"] += s["turn_count"]
            subject_stats[subj]["sessions"] += 1
            qs = s["quiz_summary"]
            subject_stats[subj]["total_quizzes"] += qs.get("total_quizzes", 0)
            subject_stats[subj]["total_questions"] += qs.get("total_questions", 0)
            subject_stats[subj]["total_correct"] += qs.get("total_correct", 0)

    for subj in subject_stats:
        qs = subject_stats[subj]
        qs["avg_accuracy"] = qs["total_correct"] / qs["total_questions"] if qs["total_questions"] > 0 else 0

    return {"sessions": sessions, "subject_stats": subject_stats}


@app.get("/api/subject-statistics")
async def subject_statistics_endpoint(subject: str = Query(..., description="学科名称")):
    """按学科聚合所有会话的学习数据"""
    tracker = get_tracker()
    record_store = get_record_store()

    matching_sessions = []
    for sid, st in tracker._trackers.items():
        if subject in st.subject_history:
            quiz_summary = record_store.get_session_summary(sid)
            # 计算该学科下的知识点掌握度
            mastery = _calculate_mastery(st, quiz_summary)
            # 匹配知识点到年级
            grade_info = _build_grade_info(st.current_topics)
            matching_sessions.append({
                "session_id": sid,
                "turn_count": st.turn_count,
                "current_topics": st.current_topics,
                "knowledge_gaps": st.knowledge_gaps,
                "mastery_levels": mastery,
                "grade_info": grade_info,
                "quiz_summary": quiz_summary,
                "last_quiz_score": st.last_quiz_score,
                "session_start_time": st.session_start_time,
            })

    matching_sessions.sort(key=lambda x: x.get("session_start_time", ""), reverse=True)

    # 聚合统计
    all_topics = {}
    for s in matching_sessions:
        for topic, score in s["mastery_levels"].items():
            if topic not in all_topics:
                all_topics[topic] = {"scores": [], "grade": s["grade_info"].get(topic, "")}
            all_topics[topic]["scores"].append(score)

    aggregated_mastery = {}
    for topic, data in all_topics.items():
        scores = data["scores"]
        aggregated_mastery[topic] = {
            "avg_score": round(sum(scores) / len(scores)) if scores else 0,
            "sessions": len(scores),
            "grade": data["grade"],
        }

    total_quizzes = sum(s["quiz_summary"].get("total_quizzes", 0) for s in matching_sessions)
    total_questions = sum(s["quiz_summary"].get("total_questions", 0) for s in matching_sessions)
    total_correct = sum(s["quiz_summary"].get("total_correct", 0) for s in matching_sessions)

    return {
        "subject": subject,
        "sessions": matching_sessions,
        "aggregated_mastery": aggregated_mastery,
        "total_quizzes": total_quizzes,
        "total_questions": total_questions,
        "total_correct": total_correct,
        "avg_accuracy": total_correct / total_questions if total_questions > 0 else 0,
    }


def _build_grade_info(topics: list) -> dict:
    """根据 KNOWLEDGE_STRUCTURE 构建知识点 → 年级的映射"""
    grade_info = {}
    for subject, grades in KNOWLEDGE_STRUCTURE.items():
        for grade, grade_topics in grades.items():
            for topic in grade_topics:
                if topic in topics:
                    grade_info[topic] = grade
    return grade_info


@app.get("/report/{session_id}", response_class=HTMLResponse)
async def report_page(session_id: str):
    """返回学情分析报告页面"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>学情分析报告 - 小智老师</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f0f2f5; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }
        .header h1 { font-size: 20px; font-weight: 600; }
        .header-actions { display: flex; gap: 12px; }
        .btn { padding: 8px 16px; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; transition: all 0.2s; }
        .btn-primary { background: white; color: #667eea; }
        .btn-primary:hover { background: #f0f0f0; }
        .btn-secondary { background: rgba(255,255,255,0.2); color: white; }
        .btn-secondary:hover { background: rgba(255,255,255,0.3); }
        .layout { display: flex; height: calc(100vh - 60px); }
        .sidebar { width: 260px; background: white; border-right: 1px solid #e5e7eb; padding: 20px; overflow-y: auto; }
        .sidebar h3 { font-size: 14px; color: #6b7280; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; justify-content: space-between; }
        .sidebar-section { margin-bottom: 24px; }
        .filter-item { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
        .filter-item:hover { background: #f3f4f6; }
        .filter-item input { accent-color: #667eea; }
        .filter-item.checked { background: #ede9fe; color: #667eea; }
        .filter-item.checked input { accent-color: #667eea; }
        .filter-clear { font-size: 12px; color: #9ca3af; cursor: pointer; text-decoration: underline; }
        .filter-clear:hover { color: #667eea; }
        .session-item { padding: 12px; border-radius: 8px; cursor: pointer; transition: all 0.2s; margin-bottom: 8px; border: 1px solid transparent; }
        .session-item:hover { background: #f3f4f6; }
        .session-item.active { background: #ede9fe; border-color: #a78bfa; }
        .session-item .session-subject { font-size: 14px; font-weight: 500; color: #1f2937; }
        .session-item .session-meta { font-size: 12px; color: #6b7280; margin-top: 4px; }
        .session-item .session-quiz { font-size: 11px; color: #10b981; margin-top: 2px; }
        .main { flex: 1; overflow-y: auto; padding: 24px; }
        .section { background: white; border-radius: 16px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        .section-title { font-size: 18px; font-weight: 600; color: #1f2937; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
        .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; }
        .stat-card .value { font-size: 32px; font-weight: 700; }
        .stat-card .label { font-size: 14px; opacity: 0.9; margin-top: 4px; }
        .mastery-list { display: flex; flex-direction: column; gap: 12px; }
        .mastery-item { display: flex; align-items: center; gap: 16px; }
        .mastery-item .topic-label { display: flex; align-items: center; gap: 8px; width: 160px; flex-shrink: 0; }
        .grade-badge { font-size: 11px; padding: 2px 6px; border-radius: 4px; background: #e0e7ff; color: #4338ca; font-weight: 500; white-space: nowrap; }
        .topic-name { font-size: 14px; color: #374151; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .mastery-bar-bg { flex: 1; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }
        .mastery-bar { height: 100%; border-radius: 4px; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); }
        .mastery-item .score { width: 40px; text-align: right; font-size: 14px; font-weight: 600; }
        .gap-list { display: flex; flex-direction: column; gap: 8px; }
        .gap-item { padding: 12px 16px; border-radius: 10px; background: #fef3c7; border-left: 4px solid #f59e0b; cursor: pointer; transition: all 0.2s; }
        .gap-item:hover { background: #fde68a; }
        .gap-item.expanded { background: #fef3c7; }
        .gap-item .gap-text { font-size: 14px; color: #92400e; }
        .gap-item .gap-detail { display: none; margin-top: 8px; padding-top: 8px; border-top: 1px solid #fde68a; font-size: 13px; color: #78350f; }
        .gap-item.expanded .gap-detail { display: block; }
        .quiz-timeline { display: flex; flex-direction: column; gap: 12px; }
        .quiz-item { display: flex; align-items: center; gap: 16px; padding: 12px; background: #f9fafb; border-radius: 10px; }
        .quiz-item .quiz-time { font-size: 12px; color: #6b7280; width: 100px; }
        .quiz-item .quiz-info { flex: 1; font-size: 14px; color: #374151; }
        .quiz-item .quiz-score { font-size: 16px; font-weight: 600; }
        .suggestion-list { display: flex; flex-direction: column; gap: 10px; }
        .suggestion-item { display: flex; gap: 12px; padding: 12px; background: #f0fdf4; border-radius: 10px; border-left: 4px solid #10b981; }
        .suggestion-item .icon { font-size: 18px; }
        .suggestion-item .text { font-size: 14px; color: #065f46; line-height: 1.6; }
        .empty-state { text-align: center; padding: 60px 20px; color: #6b7280; }
        .empty-state .icon { font-size: 48px; margin-bottom: 16px; }
        .loading { text-align: center; padding: 40px; color: #6b7280; }
        .loading .spinner { width: 40px; height: 40px; border: 3px solid #e5e7eb; border-top-color: #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 16px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .tags { display: flex; flex-wrap: wrap; gap: 8px; }
        .tag { padding: 4px 12px; border-radius: 20px; font-size: 13px; }
        .tag-subject { background: #ede9fe; color: #667eea; }
        .tag-topic { background: #dbeafe; color: #3b82f6; }
        .tag-gap { background: #fef3c7; color: #f59e0b; }
        /* 跨会话学科汇总 */
        .cross-session-summary { background: white; border-radius: 16px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        .cross-summary-title { font-size: 16px; font-weight: 600; color: #1f2937; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
        .cross-summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
        .cross-subject-card { background: linear-gradient(135deg, #f8fafc 0%, #f0f4ff 100%); border: 1px solid #e0e7ff; border-radius: 12px; padding: 16px; cursor: pointer; transition: all 0.2s; }
        .cross-subject-card:hover { border-color: #667eea; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102,126,234,0.15); }
        .cross-subject-card .subj-name { font-size: 16px; font-weight: 600; color: #1f2937; margin-bottom: 8px; }
        .cross-subject-card .subj-stats { display: flex; flex-wrap: wrap; gap: 8px; font-size: 12px; color: #6b7280; }
        .cross-subject-card .subj-stat { display: flex; align-items: center; gap: 4px; }
        .cross-subject-card .subj-stat strong { color: #667eea; }
        .filter-active-banner { background: #ede9fe; border: 1px solid #a78bfa; border-radius: 10px; padding: 10px 16px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; font-size: 13px; color: #667eea; }
        .filter-active-banner .clear-btn { background: #a78bfa; color: white; border: none; padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; }
        .filter-active-banner .clear-btn:hover { background: #667eea; }
        .hidden { display: none !important; }
    </style>
</head>
<body>
    <div class="header">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:24px;">📊</span>
            <h1>学情分析报告</h1>
        </div>
        <div class="header-actions">
            <button class="btn btn-secondary" onclick="exportReport()">📥 导出报告</button>
            <button class="btn btn-primary" onclick="window.location.href='/'">← 返回聊天</button>
        </div>
    </div>
    <div class="layout">
        <div class="sidebar">
            <div class="sidebar-section">
                <h3>学科筛选 <span class="filter-clear" onclick="toggleAllSubjects(false)">清空</span></h3>
                <div id="subject-filters"></div>
            </div>
            <div class="sidebar-section">
                <h3>会话列表 <span class="filter-clear" onclick="showAllSessions()">全部</span></h3>
                <div id="session-list"></div>
            </div>
        </div>
        <div class="main" id="main-content">
            <div class="loading"><div class="spinner"></div>正在加载报告...</div>
        </div>
    </div>
    <script>
        const sessionId = new URLSearchParams(window.location.search).get('session_id') || localStorage.getItem('sessionId') || '';
        let reportData = null;
        let allSessions = [];
        let sessionsSummary = null;
        let activeSubjects = new Set();   // 当前激活的学科
        let activeTopics = new Set();     // 当前激活的知识点
        let activeSessionId = null;       // 当前激活的会话（null = 全部）

        async function loadReport(sessionId) {
            const main = document.getElementById('main-content');
            main.innerHTML = '<div class="loading"><div class="spinner"></div>正在加载报告...</div>';
            try {
                const resp = await fetch('/api/report-data/' + sessionId);
                reportData = await resp.json();
                if (reportData.empty) {
                    main.innerHTML = '<div class="empty-state"><div class="icon">📊</div><div>暂无学习数据，多提问来生成报告吧</div></div>';
                    return;
                }
                renderReport();
            } catch (e) {
                main.innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><div>加载报告失败，请重试</div></div>';
            }
        }

        async function loadSessions() {
            try {
                const resp = await fetch('/api/sessions');
                allSessions = resp.sessions || [];
                renderSessionList();
            } catch (e) { console.error(e); }
        }

        async function loadSessionsSummary() {
            try {
                const resp = await fetch('/api/sessions-summary');
                sessionsSummary = await resp.json();
            } catch (e) { console.error(e); }
        }

        function renderSessionList() {
            const container = document.getElementById('session-list');
            if (!allSessions.length) {
                container.innerHTML = '<div style="color:#9ca3af;font-size:13px;">暂无其他会话</div>';
                return;
            }
            container.innerHTML = allSessions.map(s => {
                const isActive = activeSessionId === s.session_id || (!activeSessionId && s.session_id === sessionId);
                const quizScore = s.last_quiz_score != null ? Math.round(s.last_quiz_score * 100) + '%' : '';
                return `
                <div class="session-item ${isActive ? 'active' : ''}" onclick="switchSession('${s.session_id}')">
                    <div class="session-subject">${(s.subject_history || []).join(' / ') || '综合'}</div>
                    <div class="session-meta">${s.turn_count || 0}轮 · ${(s.current_topics || []).slice(0,2).join(', ')}</div>
                    ${quizScore ? `<div class="session-quiz">最近练习: ${quizScore}</div>` : ''}
                </div>`;
            }).join('');
        }

        function switchSession(sid) {
            activeSessionId = sid;
            const url = new URL(window.location);
            url.searchParams.set('session_id', sid);
            window.location.href = url.toString();
        }

        function showAllSessions() {
            activeSessionId = null;
            const url = new URL(window.location);
            url.searchParams.set('session_id', sessionId);
            window.location.href = url.toString();
        }

        // 初始化所有学科为选中状态
        function initSubjectFilters() {
            activeSubjects = new Set((reportData.subject_history || []).slice());
            renderSubjectFilters();
        }

        function renderSubjectFilters() {
            const subjects = reportData.subject_history || [];
            const container = document.getElementById('subject-filters');
            if (!subjects.length) {
                container.innerHTML = '<div style="color:#9ca3af;font-size:13px;">暂无学科数据</div>';
                return;
            }
            container.innerHTML = subjects.map(s => {
                const checked = activeSubjects.has(s) ? 'checked' : '';
                return `<label class="filter-item ${checked ? 'checked' : ''}">
                    <input type="checkbox" ${checked} onchange="toggleSubject('${s}')"> ${s}
                </label>`;
            }).join('');
        }

        function toggleSubject(subject) {
            if (activeSubjects.has(subject)) {
                activeSubjects.delete(subject);
            } else {
                activeSubjects.add(subject);
            }
            renderSubjectFilters();
            applyFilters();
        }

        function toggleAllSubjects(selectAll) {
            const subjects = reportData.subject_history || [];
            if (selectAll) {
                activeSubjects = new Set(subjects);
            } else {
                activeSubjects = new Set();
            }
            renderSubjectFilters();
            applyFilters();
        }

        function applyFilters() {
            renderReport();
        }

        function getScoreColor(score) {
            if (score >= 80) return '#10b981';
            if (score >= 60) return '#f59e0b';
            return '#ef4444';
        }

        function formatDate(iso) {
            if (!iso) return '-';
            const d = new Date(iso);
            return d.toLocaleDateString('zh-CN') + ' ' + d.toLocaleTimeString('zh-CN', {hour:'2-digit', minute:'2-digit'});
        }

        function renderReport() {
            const d = reportData;
            initSubjectFilters();

            // 获取筛选后的数据
            const filteredTopics = getFilteredTopics();
            const filteredMastery = getFilteredMastery();
            const filteredGaps = getFilteredGaps();

            // 显示筛选状态横幅
            let filterBanner = '';
            if (activeSubjects.size > 0 && activeSubjects.size < (d.subject_history || []).length) {
                const labels = Array.from(activeSubjects).join('、');
                filterBanner = `<div class="filter-active-banner">
                    <span>🔍 已筛选学科: <strong>${labels}</strong></span>
                    <button class="clear-btn" onclick="toggleAllSubjects(true)">清除筛选</button>
                </div>`;
            }

            // 学科标签
            const subjectTags = (d.subject_history || []).map(s => `<span class="tag tag-subject">${s}</span>`).join('');
            // 知识点标签（带年级）
            const gradeInfo = d.grade_info || {};
            const topicTags = (d.current_topics || []).map(t => {
                const grade = gradeInfo[t] ? `<span class="grade-badge">${gradeInfo[t]}</span>` : '';
                return `${grade}<span class="tag tag-topic">${t}</span>`;
            }).join('');
            // 薄弱点标签
            const gapTags = (d.knowledge_gaps || []).map(g => `<span class="tag tag-gap">${g}</span>`).join('');

            // 掌握度列表（带年级标签，支持筛选）
            const masteryHtml = Object.entries(filteredMastery).length
                ? Object.entries(filteredMastery).map(([topic, score]) => {
                    const grade = gradeInfo[topic] ? `<span class="grade-badge">${gradeInfo[topic]}</span>` : '';
                    return `
                    <div class="mastery-item">
                        <div class="topic-label">${grade}<span class="topic-name">${topic}</span></div>
                        <div class="mastery-bar-bg"><div class="mastery-bar" style="width:${score}%;background:${getScoreColor(score)}"></div></div>
                        <div class="score" style="color:${getScoreColor(score)}">${score}</div>
                    </div>`;
                }).join('')
                : '<div style="color:#9ca3af;font-size:13px;">暂无掌握度数据</div>';

            // 薄弱点详情（支持筛选）
            const gapHtml = filteredGaps.length
                ? filteredGaps.map((g, i) => `
                    <div class="gap-item" onclick="this.classList.toggle('expanded')">
                        <div class="gap-text">⚠️ ${g}</div>
                        <div class="gap-detail">💡 建议加强该知识点的练习和复习</div>
                    </div>
                `).join('')
                : '<div style="color:#10b981;font-size:14px;">🎉 暂无明显薄弱点，继续保持！</div>';

            // 练习记录
            const quizSummary = d.quiz_summary || {};
            const quizHtml = quizSummary.total_quizzes > 0
                ? `<div class="quiz-timeline">${quizSummary.records.map(r => `
                    <div class="quiz-item">
                        <div class="quiz-time">${formatDate(r.timestamp)}</div>
                        <div class="quiz-info">${r.total_count}题 · 正确${r.correct_count}题</div>
                        <div class="quiz-score" style="color:${getScoreColor(r.accuracy * 100)}">${Math.round(r.accuracy * 100)}%</div>
                    </div>
                `).join('')}</div>`
                : '<div style="color:#9ca3af;font-size:13px;">暂无练习记录</div>';

            // 改进建议
            const suggestionHtml = (d.improvement_suggestions || []).length
                ? d.improvement_suggestions.map(s => `<div class="suggestion-item"><span class="icon">💡</span><div class="text">${s}</div></div>`).join('')
                : '';

            // 优点
            const strengthHtml = (d.strengths || []).length
                ? d.strengths.map(s => `<div class="suggestion-item" style="background:#eff6ff;border-left-color:#3b82f6"><span class="icon">⭐</span><div class="text">${s}</div></div>`).join('')
                : '';

            // 弱点
            const weaknessHtml = (d.weaknesses || []).length
                ? d.weaknesses.map(s => `<div class="suggestion-item" style="background:#fff7ed;border-left-color:#f97316"><span class="icon">📌</span><div class="text">${s}</div></div>`).join('')
                : '';

            // 跨会话学科汇总
            const crossSummaryHtml = renderCrossSessionSummary();

            // 统计数字（根据筛选更新）
            const activeSubjectCount = activeSubjects.size > 0 ? activeSubjects.size : (d.subject_history || []).length;
            const activeTopicCount = filteredTopics.length;

            document.getElementById('main-content').innerHTML = `
                ${filterBanner}
                ${crossSummaryHtml}

                <!-- 概览 -->
                <div class="section">
                    <div class="section-title">📊 学习概览</div>
                    <div class="stats-grid">
                        <div class="stat-card"><div class="value">${d.turn_count || 0}</div><div class="label">对话轮次</div></div>
                        <div class="stat-card"><div class="value">${activeSubjectCount}</div><div class="label">学科数量</div></div>
                        <div class="stat-card"><div class="value">${activeTopicCount}</div><div class="label">知识点</div></div>
                        <div class="stat-card"><div class="value">${filteredGaps.length}</div><div class="label">薄弱点</div></div>
                    </div>
                    <div style="margin-top:16px">
                        <div style="font-size:14px;color:#6b7280;margin-bottom:8px">学科</div>
                        <div class="tags">${subjectTags}</div>
                    </div>
                    <div style="margin-top:12px">
                        <div style="font-size:14px;color:#6b7280;margin-bottom:8px">知识点</div>
                        <div class="tags">${topicTags}</div>
                    </div>
                    ${gapTags ? `<div style="margin-top:12px"><div style="font-size:14px;color:#6b7280;margin-bottom:8px">薄弱点</div><div class="tags">${gapTags}</div></div>` : ''}
                </div>

                <!-- 知识掌握度 -->
                <div class="section">
                    <div class="section-title">📈 知识掌握度</div>
                    <div class="mastery-list">${masteryHtml}</div>
                </div>

                <!-- 薄弱点分析 -->
                <div class="section">
                    <div class="section-title">⚠️ 薄弱点分析</div>
                    <div class="gap-list">${gapHtml}</div>
                </div>

                <!-- 练习记录 -->
                <div class="section">
                    <div class="section-title">📝 练习记录</div>
                    ${quizHtml}
                </div>

                <!-- 优点与弱点 -->
                <div class="section">
                    <div class="section-title">🎯 学习分析</div>
                    ${strengthHtml ? `<div style="margin-bottom:12px"><div style="font-size:14px;color:#6b7280;margin-bottom:8px">优点</div>${strengthHtml}</div>` : ''}
                    ${weaknessHtml ? `<div style="margin-bottom:12px"><div style="font-size:14px;color:#6b7280;margin-bottom:8px">待改进</div>${weaknessHtml}</div>` : ''}
                </div>

                <!-- 改进建议 -->
                <div class="section">
                    <div class="section-title">💡 改进建议</div>
                    ${suggestionHtml || '<div style="color:#9ca3af;font-size:13px;">暂无建议</div>'}
                    ${d.next_steps ? `<div style="margin-top:12px;padding:12px;background:#f0f9ff;border-radius:10px;font-size:14px;color:#0c4a6e"><strong>下一步建议：</strong>${d.next_steps}</div>` : ''}
                </div>

                <!-- 对话摘要 -->
                ${d.conversation_summary ? `<div class="section">
                    <div class="section-title">📋 对话摘要</div>
                    <div style="font-size:14px;color:#374151;line-height:1.8">${d.conversation_summary}</div>
                </div>` : ''}
            `;
        }

        function getFilteredTopics() {
            const topics = reportData.current_topics || [];
            if (activeSubjects.size === 0) return topics;
            // 如果没有选中学科，返回全部
            if (activeSubjects.size === (reportData.subject_history || []).length) return topics;
            // 根据当前会话数据，无法精确判断每个 topic 属于哪个学科
            // 但我们可以检查 topic 名称中是否包含学科关键词
            return topics;
        }

        function getFilteredMastery() {
            const mastery = reportData.mastery_levels || {};
            if (activeSubjects.size === 0) return mastery;
            // 当前会话级筛选，返回全部（因为 mastery 是会话维度的）
            return mastery;
        }

        function getFilteredGaps() {
            const gaps = reportData.knowledge_gaps || [];
            if (activeSubjects.size === 0) return gaps;
            return gaps;
        }

        function renderCrossSessionSummary() {
            if (!sessionsSummary || !sessionsSummary.subject_stats) return '';
            const stats = sessionsSummary.subject_stats;
            const entries = Object.entries(stats);
            if (!entries.length) return '';

            return `<div class="cross-session-summary">
                <div class="cross-summary-title">📚 跨会话学科汇总</div>
                <div class="cross-summary-grid">
                    ${entries.map(([subj, s]) => `
                        <div class="cross-subject-card" onclick="showSubjectReport('${subj}')">
                            <div class="subj-name">${subj}</div>
                            <div class="subj-stats">
                                <div class="subj-stat">会话 <strong>${s.sessions}</strong> 个</div>
                                <div class="subj-stat">轮次 <strong>${s.total_turns}</strong></div>
                                <div class="subj-stat">练习 <strong>${s.total_quizzes}</strong> 次</div>
                                <div class="subj-stat">正确率 <strong>${Math.round(s.avg_accuracy * 100)}%</strong></div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>`;
        }

        function showSubjectReport(subject) {
            // 跳转到学科统计页面
            window.open(`/api/subject-statistics?subject=${encodeURIComponent(subject)}`, '_blank');
        }

        function exportReport() {
            if (!reportData) return;
            const d = reportData;
            const lines = [
                '# 学情分析报告',
                '',
                `**生成时间**: ${new Date().toLocaleString('zh-CN')}`,
                `**会话ID**: ${sessionId}`,
                '',
                '## 学习概览',
                `- 对话轮次: ${d.turn_count || 0}`,
                `- 学科: ${(d.subject_history || []).join(', ') || '无'}`,
                `- 知识点: ${(d.current_topics || []).join(', ') || '无'}`,
                `- 薄弱点: ${(d.knowledge_gaps || []).length} 个`,
                '',
                '## 知识掌握度',
                ...Object.entries(d.mastery_levels || {}).map(([t, s]) => `- ${t}: ${s}分`),
                '',
                '## 薄弱点分析',
                ...((d.knowledge_gaps || []).map(g => `- ${g}`)),
                '',
                '## 练习记录',
                ...(d.quiz_summary?.records || []).map(r => `- ${formatDate(r.timestamp)}: ${r.correct_count}/${r.total_count} (${Math.round(r.accuracy*100)}%)`),
                '',
                '## 学习分析',
                '### 优点',
                ...((d.strengths || []).map(s => `- ${s}`)),
                '### 待改进',
                ...((d.weaknesses || []).map(s => `- ${s}`)),
                '',
                '## 改进建议',
                ...((d.improvement_suggestions || []).map(s => `- ${s}`)),
                '',
                '## 对话摘要',
                d.conversation_summary || '无',
            ];
            const blob = new Blob([lines.join('\\n')], { type: 'text/markdown;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `学情分析报告_${new Date().toISOString().slice(0,10)}.md`;
            a.click();
            URL.revokeObjectURL(url);
        }

        // 初始化
        loadSessions();
        if (sessionId) {
            loadReport(sessionId);
        } else {
            document.getElementById('main-content').innerHTML = '<div class="empty-state"><div class="icon">🔍</div><div>请先选择一个会话</div></div>';
        }
    </script>
</body>
</html>"""


@app.get("/export-note/{session_id}")
async def export_note_endpoint(
    session_id: str,
    format: str = Query("markdown", pattern="^(markdown|text)$")
):
    """导出对话笔记"""
    # 获取会话历史
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
    except Exception:
        messages = []

    # 获取 tracker 信息
    tracker = get_tracker()
    st = tracker.get_or_create(session_id)

    # 构建 Markdown 内容
    now = datetime.now().strftime("%Y-%m-%d")
    subjects = ", ".join(st.subject_history) if st.subject_history else "综合"
    topics = ", ".join(st.current_topics) if st.current_topics else "无"

    lines = [
        f"# 学习记录 - {now}",
        f"",
        f"**学科**: {subjects}",
        f"**知识点**: {topics}",
        f"**对话轮次**: {st.turn_count}",
        f"",
        f"---",
        f"",
    ]

    for msg in messages:
        if msg.type == "human":
            lines.append(f"## 学生")
            lines.append(f"{msg.content}")
            lines.append(f"")
        elif msg.type == "ai":
            lines.append(f"## 老师")
            lines.append(f"{msg.content}")
            lines.append(f"")

    content = "\n".join(lines)

    # 文件名使用 ASCII 安全格式，避免 Content-Disposition 编码问题
    safe_subjects = "".join(c if c.isascii() or c.isdigit() else "_" for c in subjects)
    filename = f"study_notes_{now}_{safe_subjects}"

    if format == "markdown":
        return Response(
            content=content,
            media_type="text/markdown;charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename={filename}.md; filename*=UTF-8''{filename}.md"
            }
        )
    else:
        return Response(
            content=content,
            media_type="text/plain;charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename={filename}.txt; filename*=UTF-8''{filename}.txt"
            }
        )


@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>小智老师 - 超级教育智能体</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f7fa; height: 100vh; display: flex; flex-direction: column; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
        .header h1 { font-size: 20px; font-weight: 600; }
        .main { flex: 1; display: flex; overflow: hidden; }
        .chat-container { flex: 1; display: flex; flex-direction: column; max-width: 900px; margin: 0 auto; width: 100%; }
        .messages { flex: 1; overflow-y: auto; padding: 20px; }
        .message { margin-bottom: 16px; display: flex; gap: 12px; }
        .message.user { flex-direction: row-reverse; }
        .message .avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
        .message.user .avatar { background: #667eea; }
        .message.assistant .avatar { background: #10b981; }
        .message .bubble { max-width: 95%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
        .message.user .bubble { background: #667eea; color: white; }
        .message.assistant .bubble { background: white; border: 1px solid #e5e7eb; }
        .input-wrapper { position: relative; }
        .input-area { padding: 16px 20px; background: white; border-top: 1px solid #e5e7eb; display: flex; gap: 12px; align-items: flex-end; }
        .input-area textarea { flex: 1; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; resize: none; font-size: 14px; min-height: 44px; max-height: 120px; font-family: inherit; }
        .input-area textarea:focus { outline: none; border-color: #667eea; }
        .input-area button { width: 44px; height: 44px; border: none; border-radius: 12px; cursor: pointer; font-size: 20px; transition: background 0.2s; flex-shrink: 0; }
        .btn-send { background: #667eea; color: white; }
        .btn-send:hover { background: #5a6fd6; }

        /* 功能按钮 */
        .btn-feature {
            width: 44px; height: 44px; border-radius: 50%;
            border: 1px solid #e5e7eb; background: #f9fafb;
            cursor: pointer; font-size: 22px;
            transition: all 0.15s ease; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
        }
        .btn-feature:hover { background: #f0f4ff; border-color: #667eea; }
        .btn-feature.active { background: #667eea; color: white; border-color: #667eea; }

        /* 功能面板 */
        .feature-panel {
            display: none;
            position: absolute; bottom: 100%; left: 0; right: 0;
            margin-bottom: 8px;
            background: white; border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.12);
            padding: 16px; max-height: 60vh; overflow-y: auto;
            z-index: 100;
        }
        .feature-panel.active {
            display: block;
            animation: panelSlideUp 0.15s ease;
        }
        @keyframes panelSlideUp {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .panel-group { margin-bottom: 16px; }
        .panel-group:last-child { margin-bottom: 0; }
        .panel-group-title {
            font-size: 12px; color: #9ca3af; font-weight: 500;
            margin-bottom: 8px; padding-left: 4px;
        }
        .panel-divider { border-top: 1px solid #e5e7eb; margin: 12px 0; }

        .feature-item {
            display: flex; align-items: center; gap: 12px;
            padding: 10px 12px; border-radius: 10px;
            cursor: pointer; transition: background 0.15s;
        }
        .feature-item:hover { background: #f0f4ff; }
        .feature-item .icon { font-size: 22px; width: 32px; text-align: center; }
        .feature-item .text { flex: 1; }
        .feature-item .title { font-size: 14px; font-weight: 500; color: #374151; }
        .feature-item .subtitle { font-size: 12px; color: #9ca3af; margin-top: 2px; }
        .feature-item .arrow { color: #d1d5db; font-size: 16px; }

        /* 图片预览 */
        .image-preview-bar {
            display: none; padding: 8px 20px;
            background: white; border-top: 1px solid #e5e7eb;
            align-items: center; gap: 8px;
        }
        .image-preview-bar.active { display: flex; }
        .image-preview-thumb {
            width: 48px; height: 48px; border-radius: 8px;
            object-fit: cover; border: 1px solid #e5e7eb;
        }
        .image-preview-name { flex: 1; font-size: 13px; color: #6b7280; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .image-preview-remove {
            width: 28px; height: 28px; border-radius: 50%;
            border: none; background: #fee2e2; color: #ef4444;
            cursor: pointer; font-size: 14px;
            display: flex; align-items: center; justify-content: center;
        }
        .image-preview-remove:hover { background: #fecaca; }

        /* 试卷卡片 */
        .quiz-card {
            background: white; border: 1px solid #e5e7eb;
            border-radius: 12px; padding: 16px; margin-top: 8px;
        }
        .quiz-card .question { font-size: 15px; font-weight: 500; color: #1f2937; margin-bottom: 12px; }
        .quiz-card .option {
            display: block; padding: 8px 12px; margin: 4px 0;
            border: 1px solid #e5e7eb; border-radius: 8px;
            cursor: pointer; font-size: 14px; transition: all 0.15s;
        }
        .quiz-card .option:hover { background: #f0f4ff; border-color: #667eea; }
        .quiz-card .option.selected { background: #667eea; color: white; border-color: #667eea; }
        .quiz-card .option.correct { background: #d1fae5; border-color: #10b981; }
        .quiz-card .option.wrong { background: #fee2e2; border-color: #ef4444; }
        .quiz-card .submit-btn {
            margin-top: 12px; padding: 8px 20px; border: none;
            border-radius: 8px; background: #667eea; color: white;
            cursor: pointer; font-size: 14px;
        }
        .quiz-card .submit-btn:hover { background: #5a6fd6; }
        .quiz-card .submit-btn:disabled { background: #9ca3af; cursor: not-allowed; }
        .quiz-card .explanation {
            margin-top: 12px; padding: 12px; background: #f0f9ff;
            border-radius: 8px; font-size: 13px; color: #0c4a6e;
            border-left: 3px solid #0ea5e9;
        }
        .quiz-card .score-display {
            margin-top: 12px; padding: 12px; background: #f0fdf4;
            border-radius: 8px; font-size: 15px; font-weight: 600;
            color: #166534; text-align: center;
        }
        /* 填空题样式 */
        .fill-input {
            width: 100%; padding: 8px 12px; border: 1px solid #e5e7eb;
            border-radius: 8px; font-size: 14px; margin-top: 8px;
            transition: border-color 0.15s;
        }
        .fill-input:focus { outline: none; border-color: #667eea; }
        .fill-input:disabled { background: #f9fafb; cursor: default; }
        .fill-input.correct { border-color: #10b981; background: #d1fae5; }
        .fill-input.wrong { border-color: #ef4444; background: #fee2e2; }
        .fill-answer { margin-top: 8px; font-size: 13px; color: #6b7280; }
        .fill-answer strong { color: #1f2937; }

        /* 学习报告 */
        .report-panel {
            background: white; border: 1px solid #e5e7eb;
            border-radius: 12px; padding: 16px; margin-top: 8px;
        }
        .report-panel .empty-state {
            text-align: center; padding: 24px; color: #9ca3af;
        }
        .report-panel .empty-state .icon { font-size: 32px; margin-bottom: 8px; }
        .report-stat {
            display: flex; justify-content: space-around;
            padding: 12px 0; border-bottom: 1px solid #f3f4f6;
        }
        .report-stat:last-child { border-bottom: none; }
        .report-stat .label { font-size: 12px; color: #9ca3af; }
        .report-stat .value { font-size: 20px; font-weight: 600; color: #667eea; }
        .report-topics { margin-top: 12px; }
        .report-topics .label { font-size: 13px; font-weight: 500; color: #374151; margin-bottom: 6px; }
        .report-topic {
            display: inline-block; padding: 4px 10px; margin: 4px 4px 4px 0;
            background: #f0f4ff; color: #4338ca; border-radius: 12px; font-size: 12px;
        }
        .report-gap {
            display: inline-block; padding: 4px 10px; margin: 4px 4px 4px 0;
            background: #fef3c7; color: #92400e; border-radius: 12px; font-size: 12px;
        }

        .mindmap-section { margin-top: 12px; padding: 12px; background: #f0f4ff; border-radius: 10px; border: 1px solid #c7d2fe; cursor: zoom-in; position: relative; transition: background 0.2s; }
        .mindmap-section:hover { background: #e0eaff; }
        .mindmap-label { font-size: 13px; font-weight: 600; color: #4338ca; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }
        .mindmap-label .zoom-hint { font-size: 11px; color: #818cf8; font-weight: 400; margin-left: auto; }
        .mindmap-mermaid { overflow-x: auto; }
        .mindmap-mermaid .mermaid { display: flex; justify-content: center; }
        .mindmap-download { margin-top: 8px; padding: 6px 12px; border: 1px solid #c7d2fe; border-radius: 6px; background: white; cursor: pointer; font-size: 12px; color: #4338ca; transition: background 0.2s; }
        .mindmap-download:hover { background: #e0eaff; }
        .loading { color: #9ca3af; font-style: italic; }
        .context-badge { display: inline-block; background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-top: 8px; }

        /* 缩放模态框 */
        .zoom-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; justify-content: center; align-items: center; }
        .zoom-overlay.active { display: flex; }
        .zoom-container { position: relative; width: 90vw; height: 90vh; background: white; border-radius: 16px; overflow: hidden; display: flex; flex-direction: column; }
        .zoom-toolbar { display: flex; align-items: center; gap: 8px; padding: 12px 16px; border-bottom: 1px solid #e5e7eb; background: #f9fafb; flex-shrink: 0; }
        .zoom-toolbar .title { font-size: 14px; font-weight: 600; color: #374151; flex: 1; }
        .zoom-toolbar button { padding: 6px 12px; border: 1px solid #d1d5db; border-radius: 6px; background: white; cursor: pointer; font-size: 13px; transition: background 0.2s; }
        .zoom-toolbar button:hover { background: #f3f4f6; }
        .zoom-toolbar .btn-close { color: #ef4444; border-color: #fca5a5; }
        .zoom-toolbar .btn-close:hover { background: #fef2f2; }
        .zoom-body { flex: 1; overflow: hidden; position: relative; }
        .zoom-body .mermaid-wrap { width: 100%; height: 100%; display: flex; justify-content: center; align-items: flex-start; overflow: auto; cursor: grab; }
        .zoom-body .mermaid-wrap:active { cursor: grabbing; }
        .zoom-body .mermaid-wrap svg { max-width: none; transition: transform 0.15s ease; transform-origin: top center; }

        /* 导出下拉 */
        .export-dropdown { position: relative; display: inline-block; }
        .export-menu {
            display: none; position: absolute; bottom: 100%; left: 0; margin-bottom: 8px;
            background: white; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            padding: 8px; z-index: 200; min-width: 120px;
        }
        .export-menu.active { display: block; }
        .export-menu button {
            width: 100%; padding: 8px 12px; border: none; background: none;
            text-align: left; cursor: pointer; border-radius: 6px; font-size: 13px;
        }
        .export-menu button:hover { background: #f0f4ff; }

        @media (max-width: 768px) {
            .chat-container { padding: 0; }
            .message .bubble { max-width: 100%; }
        }
    </style>
</head>
<body>
    <div class="header">
        <span style="font-size:24px">🎓</span>
        <h1>小智老师</h1>
    </div>
    <div class="main">
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="avatar">🤖</div>
                    <div class="bubble">你好！我是小智老师，你的初中学习助手。请输入你的问题。</div>
                </div>
            </div>
            <!-- 图片预览栏 -->
            <div class="image-preview-bar" id="imagePreviewBar">
                <img class="image-preview-thumb" id="imagePreviewThumb" src="" alt="预览">
                <span class="image-preview-name" id="imagePreviewName"></span>
                <button class="image-preview-remove" id="imagePreviewRemove">✕</button>
            </div>
            <!-- 输入区域 -->
            <div class="input-wrapper">
                <div class="input-area">
                    <button class="btn-feature" id="featureBtn" title="功能菜单">⊕</button>
                    <textarea id="userInput" placeholder="输入你的问题..." rows="1"></textarea>
                    <button class="btn-send" id="sendBtn">➤</button>
                </div>
                <!-- 功能面板 -->
                <div class="feature-panel" id="featurePanel">
                    <div class="panel-group">
                        <div class="panel-group-title">输入与学习</div>
                        <div class="feature-item" data-action="image">
                            <span class="icon">📷</span>
                            <div class="text">
                                <div class="title">图片解析</div>
                                <div class="subtitle">上传题目图片，AI 自动识别讲解</div>
                            </div>
                            <span class="arrow">›</span>
                        </div>
                        <div class="feature-item" data-action="mindmap">
                            <span class="icon">🧠</span>
                            <div class="text">
                                <div class="title">创建思维导图</div>
                                <div class="subtitle">基于当前对话生成可交互知识图谱</div>
                            </div>
                            <span class="arrow">›</span>
                        </div>
                        <div class="feature-item" data-action="quiz">
                            <span class="icon">📝</span>
                            <div class="text">
                                <div class="title">自动出卷</div>
                                <div class="subtitle">基于当前知识点生成练习题</div>
                            </div>
                            <span class="arrow">›</span>
                        </div>
                    </div>
                    <div class="panel-divider"></div>
                    <div class="panel-group">
                        <div class="panel-group-title">数据与沉淀</div>
                        <div class="feature-item" data-action="report">
                            <span class="icon">📊</span>
                            <div class="text">
                                <div class="title">学习报告</div>
                                <div class="subtitle">查看学习频次、薄弱知识点趋势</div>
                            </div>
                            <span class="arrow">›</span>
                        </div>
                        <div class="feature-item" data-action="export">
                            <span class="icon">📥</span>
                            <div class="text">
                                <div class="title">导出笔记</div>
                                <div class="subtitle">导出当前对话为 Markdown</div>
                            </div>
                            <span class="arrow">›</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <!-- 缩放模态框 -->
    <div class="zoom-overlay" id="zoomOverlay">
        <div class="zoom-container">
            <div class="zoom-toolbar">
                <span class="title">📊 知识点思维导图</span>
                <button onclick="zoomOut()">➖ 缩小</button>
                <button onclick="zoomReset()">↺ 重置</button>
                <button onclick="zoomIn()">➕ 放大</button>
                <button class="btn-close" onclick="closeZoom()">✕ 关闭</button>
            </div>
            <div class="zoom-body">
                <div class="mermaid-wrap" id="zoomWrap">
                    <div class="mermaid" id="zoomContent"></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        mermaid.initialize({ startOnLoad: true, theme: 'default' });
        let sessionId = localStorage.getItem('sessionId') || '';
        if (!sessionId) { sessionId = crypto.randomUUID(); localStorage.setItem('sessionId', sessionId); }

        const messagesEl = document.getElementById('messages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const featureBtn = document.getElementById('featureBtn');
        const featurePanel = document.getElementById('featurePanel');
        const imagePreviewBar = document.getElementById('imagePreviewBar');
        const imagePreviewThumb = document.getElementById('imagePreviewThumb');
        const imagePreviewName = document.getElementById('imagePreviewName');
        const imagePreviewRemove = document.getElementById('imagePreviewRemove');
        const imageInput = document.createElement('input');
        imageInput.type = 'file';
        imageInput.accept = 'image/*';
        imageInput.style.display = 'none';
        document.body.appendChild(imageInput);
        const zoomOverlay = document.getElementById('zoomOverlay');
        const zoomWrap = document.getElementById('zoomWrap');
        const zoomContent = document.getElementById('zoomContent');

        let zoomLevel = 1;
        let currentMermaidCode = '';
        let pendingImageBase64 = null;
        let pendingImageName = '';

        // ============ 功能面板 ============
        featureBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleFeaturePanel();
        });

        function toggleFeaturePanel() {
            const isActive = featurePanel.classList.contains('active');
            if (isActive) {
                closeFeaturePanel();
            } else {
                featurePanel.classList.add('active');
                featureBtn.classList.add('active');
                featureBtn.textContent = '✕';
            }
        }

        function closeFeaturePanel() {
            featurePanel.classList.remove('active');
            featureBtn.classList.remove('active');
            featureBtn.textContent = '⊕';
        }

        // 点击面板外部关闭
        document.addEventListener('click', (e) => {
            if (!featurePanel.contains(e.target) && e.target !== featureBtn) {
                closeFeaturePanel();
            }
        });

        // 面板内功能项点击
        featurePanel.addEventListener('click', (e) => {
            const item = e.target.closest('.feature-item');
            if (!item) return;
            const action = item.dataset.action;
            closeFeaturePanel();
            handleFeatureAction(action);
        });

        function handleFeatureAction(action) {
            switch(action) {
                case 'image':
                    imageInput.click();
                    break;
                case 'mindmap':
                    triggerMindmap();
                    break;
                case 'quiz':
                    generateQuiz();
                    break;
                case 'report':
                    showStudyReport();
                    break;
                case 'export':
                    exportNotes();
                    break;
            }
        }

        // ============ 图片上传 ============
        imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            pendingImageName = file.name;
            const reader = new FileReader();
            reader.onload = (ev) => {
                pendingImageBase64 = ev.target.result.split(',')[1];
                imagePreviewThumb.src = ev.target.result;
                imagePreviewName.textContent = file.name;
                imagePreviewBar.classList.add('active');
            };
            reader.readAsDataURL(file);
            imageInput.value = '';
        });

        imagePreviewRemove.addEventListener('click', () => {
            pendingImageBase64 = null;
            pendingImageName = '';
            imagePreviewBar.classList.remove('active');
            imagePreviewThumb.src = '';
        });

        // ============ 发送消息 ============
        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text && !pendingImageBase64) return;

            // 如果有图片但没有文字，用图片识别内容作为消息
            let sendText = text;
            let imageB64 = pendingImageBase64;

            if (sendText && imageB64) {
                // 有文字也有图片，图片先上传获取识别文本
                try {
                    const formData = new FormData();
                    const blob = base64ToBlob(imageB64);
                    formData.append('file', blob, pendingImageName);
                    formData.append('session_id', sessionId);
                    const uploadResp = await fetch('/upload-image', { method: 'POST', body: formData });
                    const uploadData = await uploadResp.json();
                    sendText = uploadData.recognized_text + (text ? '\\n\\n' + text : '');
                    imageB64 = null;
                } catch (e) {
                    console.error('图片上传失败:', e);
                }
            } else if (imageB64) {
                // 只有图片
                try {
                    const formData = new FormData();
                    const blob = base64ToBlob(imageB64);
                    formData.append('file', blob, pendingImageName);
                    formData.append('session_id', sessionId);
                    const uploadResp = await fetch('/upload-image', { method: 'POST', body: formData });
                    const uploadData = await uploadResp.json();
                    sendText = '[图片识别内容]\\n' + uploadData.recognized_text + '\\n\\n请帮我讲解以上内容。';
                    imageB64 = null;
                } catch (e) {
                    console.error('图片上传失败:', e);
                    addMessage('assistant', '图片上传失败，请重试。');
                    return;
                }
            }

            // 清除图片状态
            pendingImageBase64 = null;
            pendingImageName = '';
            imagePreviewBar.classList.remove('active');
            imagePreviewThumb.src = '';

            addMessage('user', sendText);
            const loadingEl = document.createElement('div');
            loadingEl.className = 'message assistant';
            loadingEl.id = 'loading-msg';
            loadingEl.innerHTML = '<div class="avatar">🤖</div><div class="bubble loading">小智老师正在思考...</div>';
            messagesEl.appendChild(loadingEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId, user_input: sendText, image_base64: imageB64 || ''})
                });
                const data = await response.json();
                document.getElementById('loading-msg')?.remove();
                addMessage('assistant', data.answer, {
                    mindmap: data.mindmap_mermaid,
                    context: data.context
                });
            } catch (error) {
                document.getElementById('loading-msg')?.remove();
                addMessage('assistant', `抱歉，出错了：${error.message}`);
            }
            userInput.value = '';
            userInput.style.height = 'auto';
        }

        function base64ToBlob(base64, mimeType = 'image/jpeg') {
            const bytes = atob(base64);
            const ab = new ArrayBuffer(bytes.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < bytes.length; i++) {
                ia[i] = bytes.charCodeAt(i);
            }
            return new Blob([ab], { type: mimeType });
        }

        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        });
        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
        });

        // ============ 消息渲染 ============
        function addMessage(role, content, metadata = {}) {
            const msgEl = document.createElement('div');
            msgEl.className = `message ${role}`;
            let bubbleContent = content;
            if (metadata.mindmap && metadata.mindmap.trim()) {
                setTimeout(() => renderMindmapInline(msgEl, metadata.mindmap), 100);
            }
            let contextBadge = '';
            if (metadata.context && metadata.context.turn_count > 0) {
                contextBadge = `<div class="context-badge">第${metadata.context.turn_count}轮对话</div>`;
            }
            msgEl.innerHTML = `
                <div class="avatar">${role === 'user' ? '👤' : '🤖'}</div>
                <div class="bubble">${bubbleContent}${contextBadge}</div>
            `;
            messagesEl.appendChild(msgEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        function renderMindmapInline(msgEl, mermaidCode) {
            if (!mermaidCode || mermaidCode.trim() === '') return;
            const bubble = msgEl.querySelector('.bubble');
            const section = document.createElement('div');
            section.className = 'mindmap-section';
            section.innerHTML = `
                <div class="mindmap-label">
                    📊 知识点思维导图
                    <span class="zoom-hint">点击放大</span>
                </div>
            `;
            const mermaidDiv = document.createElement('div');
            mermaidDiv.className = 'mindmap-mermaid';
            mermaidDiv.innerHTML = `<div class="mermaid">${mermaidCode}</div>`;
            section.appendChild(mermaidDiv);

            const downloadBtn = document.createElement('button');
            downloadBtn.className = 'mindmap-download';
            downloadBtn.textContent = '📥 下载 SVG';
            downloadBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                downloadMindmap(mermaidCode);
            });
            section.appendChild(downloadBtn);

            bubble.appendChild(section);
            section.addEventListener('click', (e) => {
                if (e.target === downloadBtn || e.target.closest('.mindmap-download')) return;
                openZoom(mermaidCode);
            });

            try {
                mermaid.run({ nodes: mermaidDiv.querySelectorAll('.mermaid') });
            } catch (e) {
                mermaidDiv.innerHTML = `<pre>${mermaidCode}</pre>`;
            }
        }

        // ============ 功能：创建思维导图 ============
        function triggerMindmap() {
            addMessage('user', '请根据当前对话生成知识点思维导图');
            showLoading();
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: sessionId, user_input: '请根据当前对话生成知识点思维导图'})
            }).then(r => r.json()).then(data => {
                hideLoading();
                addMessage('assistant', data.answer, {
                    mindmap: data.mindmap_mermaid,
                    context: data.context
                });
            }).catch(err => {
                hideLoading();
                addMessage('assistant', '生成思维导图失败，请重试。');
            });
        }

        // ============ 功能：自动出卷 ============
        let currentQuizState = { questions: [], answers: {}, total: 0, submitted: 0 };

        async function generateQuiz() {
            const loadingEl = document.createElement('div');
            loadingEl.className = 'message assistant';
            loadingEl.innerHTML = '<div class="avatar">🤖</div><div class="bubble loading">正在生成试卷...</div>';
            messagesEl.appendChild(loadingEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;

            try {
                const response = await fetch('/generate-quiz', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId, count: 5})
                });
                const data = await response.json();
                loadingEl.remove();
                addMessage('assistant', `已为你生成 ${data.questions.length} 道练习题，请作答：`);
                // 重置答题状态
                currentQuizState = { questions: data.questions, answers: {}, total: data.questions.length, submitted: 0 };
                renderQuizCard(data.questions);
            } catch (error) {
                loadingEl.remove();
                addMessage('assistant', '生成试卷失败，请重试。');
            }
        }

        function renderQuizCard(questions) {
            const msgEl = document.createElement('div');
            msgEl.className = 'message assistant';
            msgEl.innerHTML = `<div class="avatar">🤖</div><div class="bubble"></div>`;
            messagesEl.appendChild(msgEl);

            const bubble = msgEl.querySelector('.bubble');

            questions.forEach((q, idx) => {
                const card = document.createElement('div');
                card.className = 'quiz-card';
                card.dataset.index = idx;
                card.dataset.correct = q.correct_answer || '';
                card.dataset.type = q.type || 'choice';

                let answerArea = '';
                if ((q.type || 'choice') === 'fill') {
                    answerArea = `
                        <input type="text" class="fill-input"
                            placeholder="请输入答案..."
                            onkeydown="if(event.key==='Enter')submitQuestion(this.closest('.submit-btn'))">
                        <div class="fill-answer" style="display:none"></div>`;
                } else {
                    let optionsHtml = '';
                    if (q.options && q.options.length > 0) {
                        optionsHtml = q.options.map((opt, i) => {
                            const label = opt.charAt(0);
                            return `<div class="option" data-value="${label}" onclick="selectOption(this)">${opt}</div>`;
                        }).join('');
                    }
                    answerArea = `<div class="options">${optionsHtml}</div>`;
                }

                card.innerHTML = `
                    <div class="question">${idx + 1}. ${q.question}</div>
                    ${answerArea}
                    <button class="submit-btn" onclick="submitQuestion(this, ${idx})">提交答案</button>
                    <div class="explanation" style="display:none">${q.explanation || ''}</div>
                `;
                bubble.appendChild(card);
            });

            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        function selectOption(el) {
            const card = el.closest('.quiz-card');
            card.querySelectorAll('.option').forEach(o => o.classList.remove('selected'));
            el.classList.add('selected');
        }

        function submitQuestion(btn, idx) {
            const card = btn.closest('.quiz-card');
            const q = currentQuizState.questions[idx];
            const correct = card.dataset.correct;

            // 填空题
            if (q.type === 'fill') {
                const input = card.querySelector('.fill-input');
                const answer = input.value.trim();
                if (!answer) {
                    alert('请先填写答案');
                    return;
                }
                const isCorrect = answer.toLowerCase() === correct.toLowerCase();
                input.disabled = true;
                input.classList.add(isCorrect ? 'correct' : 'wrong');
                const fillAnswerEl = card.querySelector('.fill-answer');
                fillAnswerEl.style.display = 'block';
                fillAnswerEl.innerHTML = isCorrect
                    ? `<span style="color:#166534">✓ 正确</span>`
                    : `<span style="color:#991b1b">✗ 你的答案：${answer}，正确答案：<strong>${correct}</strong></span>`;
                btn.disabled = true;
                btn.textContent = '已提交';
                card.querySelector('.explanation').style.display = 'block';
                currentQuizState.answers[idx] = answer;
            }
            // 选择题
            else {
                const selected = card.querySelector('.option.selected');
                if (!selected) {
                    alert('请先选择一个答案');
                    return;
                }
                const selectedValue = selected.dataset.value;
                const isCorrect = selectedValue === correct;

                card.querySelectorAll('.option').forEach(opt => {
                    opt.style.cursor = 'default';
                    opt.onclick = null;
                    if (opt.dataset.value === correct) {
                        opt.classList.add('correct');
                    } else if (opt.classList.contains('selected') && !isCorrect) {
                        opt.classList.add('wrong');
                    }
                });

                btn.disabled = true;
                btn.textContent = '已提交';
                card.querySelector('.explanation').style.display = 'block';
                currentQuizState.answers[idx] = selectedValue;
            }

            currentQuizState.submitted++;

            // 所有题目提交完 → 回传成绩
            if (currentQuizState.submitted >= currentQuizState.total) {
                submitQuizResult();
            }
        }

        async function submitQuizResult() {
            const totalCorrect = Object.entries(currentQuizState.answers).filter(
                ([idx, ans]) => ans === currentQuizState.questions[idx].correct_answer
            ).length;

            try {
                const resp = await fetch('/submit-quiz', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: sessionId,
                        questions: currentQuizState.questions,
                        answers: currentQuizState.answers,
                        correct_count: totalCorrect,
                    })
                });
                const result = await resp.json();
                // 显示总成绩卡片
                showScoreCard(result);
            } catch (e) {
                // 静默失败，不影响已提交的答题体验
                console.error('提交成绩失败:', e);
            }
        }

        function showScoreCard(result) {
            const msgEl = document.createElement('div');
            msgEl.className = 'message assistant';
            const pct = Math.round(result.accuracy * 100);
            const emoji = pct >= 80 ? '🎉' : pct >= 60 ? '👍' : '💪';
            msgEl.innerHTML = `<div class="avatar">🤖</div><div class="bubble"><div class="score-display">${emoji} 本次练习得分：${result.correct_count}/${result.total_count}（${pct}%）</div></div>`;
            messagesEl.appendChild(msgEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        // ============ 功能：学习报告 ============
        function showStudyReport() {
            if (!sessionId) {
                addMessage('assistant', '请先开始对话，生成学习数据后再查看报告。');
                return;
            }
            window.location.href = `/report/${sessionId}`;
        }

        // ============ 功能：导出笔记 ============
        function exportNotes() {
            const url = `/export-note/${sessionId}?format=markdown`;
            const a = document.createElement('a');
            a.href = url;
            a.download = '';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }

        // ============ 缩放功能 ============
        function openZoom(mermaidCode) {
            currentMermaidCode = mermaidCode;
            zoomLevel = 1;
            zoomContent.innerHTML = `<div class="mermaid">${mermaidCode}</div>`;
            zoomOverlay.classList.add('active');
            zoomWrap.style.transform = '';
            zoomWrap.style.transformOrigin = 'top left';
            try {
                mermaid.run({ nodes: zoomContent.querySelectorAll('.mermaid') });
            } catch (e) {}
            setTimeout(() => applyZoom(), 100);
        }

        function closeZoom() {
            zoomOverlay.classList.remove('active');
        }

        function applyZoom() {
            const svg = zoomContent.querySelector('svg');
            if (!svg) return;
            const wrapW = zoomWrap.clientWidth;
            const wrapH = zoomWrap.clientHeight;
            const svgW = svg.getAttribute('width');
            const svgH = svg.getAttribute('height');
            if (svgW && svgH) {
                const scale = Math.min(
                    Math.min(wrapW / parseFloat(svgW), 1) * 1.5,
                    Math.min(wrapH / parseFloat(svgH), 1) * 1.5
                );
                const finalScale = Math.max(0.8, Math.min(scale, 2.5));
                zoomLevel = finalScale;
                svg.style.transform = `scale(${zoomLevel})`;
                svg.style.transformOrigin = 'top center';
                svg.style.display = 'block';
            }
        }

        function zoomIn() {
            zoomLevel = Math.min(zoomLevel * 1.3, 4);
            const svg = zoomContent.querySelector('svg');
            if (svg) svg.style.transform = `scale(${zoomLevel})`;
        }

        function zoomOut() {
            zoomLevel = Math.max(zoomLevel / 1.3, 0.3);
            const svg = zoomContent.querySelector('svg');
            if (svg) svg.style.transform = `scale(${zoomLevel})`;
        }

        function zoomReset() {
            zoomLevel = 1;
            const svg = zoomContent.querySelector('svg');
            if (svg) svg.style.transform = 'scale(1)';
        }

        zoomWrap.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (e.deltaY < 0) zoomIn();
            else zoomOut();
        }, { passive: false });

        zoomOverlay.addEventListener('click', (e) => {
            if (e.target === zoomOverlay) closeZoom();
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeZoom();
        });

        let isDragging = false, dragStartX, dragStartY, scrollStartX, scrollStartY;
        zoomWrap.addEventListener('mousedown', (e) => {
            if (e.target.closest('.mermaid')) {
                isDragging = true;
                dragStartX = e.clientX;
                dragStartY = e.clientY;
                scrollStartX = zoomWrap.scrollLeft;
                scrollStartY = zoomWrap.scrollTop;
                zoomWrap.style.cursor = 'grabbing';
            }
        });
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            e.preventDefault();
            zoomWrap.scrollLeft = scrollStartX - (e.clientX - dragStartX);
            zoomWrap.scrollTop = scrollStartY - (e.clientY - dragStartY);
        });
        document.addEventListener('mouseup', () => {
            isDragging = false;
            zoomWrap.style.cursor = 'grab';
        });

        function downloadMindmap(mermaidCode) {
            const mermaidDiv = document.createElement('div');
            mermaidDiv.innerHTML = `<div class="mermaid">${mermaidCode}</div>`;
            document.body.appendChild(mermaidDiv);
            mermaid.run({ nodes: mermaidDiv.querySelectorAll('.mermaid') }).then(() => {
                const svg = mermaidDiv.querySelector('svg');
                if (svg) {
                    const svgData = new XMLSerializer().serializeToString(svg);
                    const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = '知识点思维导图.svg';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }
            }).catch(() => {
                const blob = new Blob([mermaidCode], { type: 'text/plain;charset=utf-8' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = '知识点思维导图.mmd';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            });
            document.body.removeChild(mermaidDiv);
        }

        function showLoading() {
            const loadingEl = document.createElement('div');
            loadingEl.className = 'message assistant';
            loadingEl.id = 'loading-msg';
            loadingEl.innerHTML = '<div class="avatar">🤖</div><div class="bubble loading">小智老师正在思考...</div>';
            messagesEl.appendChild(loadingEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        function hideLoading() {
            document.getElementById('loading-msg')?.remove();
        }
    </script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
