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
from agent.prompts import QUIZ_BATCH_PROMPT

load_dotenv()


class ChatRequest(BaseModel):
    session_id: str = ""
    user_input: str
    image_base64: str = ""


class QuizRequest(BaseModel):
    session_id: str = ""
    count: int = 5


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

                let optionsHtml = '';
                if (q.options && q.options.length > 0) {
                    optionsHtml = q.options.map((opt, i) => {
                        const label = opt.charAt(0);
                        return `<div class="option" data-value="${label}" onclick="selectOption(this)">${opt}</div>`;
                    }).join('');
                }

                card.innerHTML = `
                    <div class="question">${idx + 1}. ${q.question}</div>
                    <div class="options">${optionsHtml}</div>
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
            const selected = card.querySelector('.option.selected');
            const correct = card.dataset.correct;

            if (!selected) {
                alert('请先选择一个答案');
                return;
            }

            const selectedValue = selected.dataset.value;
            const isCorrect = selectedValue === correct;

            // 显示正确/错误
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
        }

        // ============ 功能：学习报告 ============
        async function showStudyReport() {
            const loadingEl = document.createElement('div');
            loadingEl.className = 'message assistant';
            loadingEl.innerHTML = '<div class="avatar">🤖</div><div class="bubble loading">正在生成学习报告...</div>';
            messagesEl.appendChild(loadingEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;

            try {
                const response = await fetch(`/study-report/${sessionId}`);
                const data = await response.json();
                loadingEl.remove();

                const msgEl = document.createElement('div');
                msgEl.className = 'message assistant';
                msgEl.innerHTML = `<div class="avatar">🤖</div><div class="bubble"></div>`;
                messagesEl.appendChild(msgEl);

                const bubble = msgEl.querySelector('.bubble');
                const panel = document.createElement('div');
                panel.className = 'report-panel';

                if (data.empty) {
                    panel.innerHTML = `<div class="empty-state"><div class="icon">📊</div><div>${data.message}</div></div>`;
                } else {
                    let topicsHtml = (data.current_topics || []).map(t => `<span class="report-topic">${t}</span>`).join('');
                    let gapsHtml = (data.knowledge_gaps || []).map(g => `<span class="report-gap">${g}</span>`).join('');
                    let subjectsHtml = (data.subject_history || []).map(s => `<span class="report-topic">${s}</span>`).join('');

                    panel.innerHTML = `
                        <div class="report-stat">
                            <div><div class="value">${data.turn_count}</div><div class="label">对话轮次</div></div>
                            <div><div class="value">${subjectsHtml || '-'}</div><div class="label">学科</div></div>
                        </div>
                        ${topicsHtml ? `<div class="report-topics"><div class="label">当前知识点</div>${topicsHtml}</div>` : ''}
                        ${gapsHtml ? `<div class="report-topics"><div class="label">可能薄弱的地方</div>${gapsHtml}</div>` : ''}
                        ${data.conversation_summary ? `<div class="report-topics"><div class="label">对话摘要</div><div style="font-size:13px;color:#6b7280;margin-top:4px">${data.conversation_summary}</div></div>` : ''}
                    `;
                }

                bubble.appendChild(panel);
                messagesEl.scrollTop = messagesEl.scrollHeight;
            } catch (error) {
                loadingEl.remove();
                addMessage('assistant', '获取学习报告失败，请重试。');
            }
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
