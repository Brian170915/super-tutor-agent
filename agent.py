"""
超级教育智能体 - 主入口（带上下文管理）
"""
import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langserve import add_routes

from agent.graph import build_graph
from rag.knowledge_structure import KNOWLEDGE_STRUCTURE, get_subjects, get_grades, get_topics
from agent.context_tracker import get_tracker

load_dotenv()


class ChatRequest(BaseModel):
    session_id: str = ""
    user_input: str


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
        .chat-container { flex: 1; display: flex; flex-direction: column; max-width: 800px; margin: 0 auto; width: 100%; }
        .messages { flex: 1; overflow-y: auto; padding: 20px; }
        .message { margin-bottom: 16px; display: flex; gap: 12px; }
        .message.user { flex-direction: row-reverse; }
        .message .avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
        .message.user .avatar { background: #667eea; }
        .message.assistant .avatar { background: #10b981; }
        .message .bubble { max-width: 70%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
        .message.user .bubble { background: #667eea; color: white; }
        .message.assistant .bubble { background: white; border: 1px solid #e5e7eb; }
        .input-area { padding: 16px 20px; background: white; border-top: 1px solid #e5e7eb; display: flex; gap: 12px; }
        .input-area textarea { flex: 1; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; resize: none; font-size: 14px; min-height: 44px; max-height: 120px; font-family: inherit; }
        .input-area textarea:focus { outline: none; border-color: #667eea; }
        .input-area button { width: 44px; height: 44px; border: none; border-radius: 12px; cursor: pointer; font-size: 20px; transition: background 0.2s; }
        .btn-send { background: #667eea; color: white; }
        .btn-send:hover { background: #5a6fd6; }
        .mindmap-panel { width: 400px; background: white; border-left: 1px solid #e5e7eb; display: none; flex-direction: column; overflow: hidden; }
        .mindmap-panel.active { display: flex; }
        .mindmap-panel .panel-header { padding: 16px; border-bottom: 1px solid #e5e7eb; font-weight: 600; }
        .mindmap-panel .panel-body { flex: 1; overflow: auto; padding: 16px; }
        .mindmap-panel .panel-body .mermaid { display: flex; justify-content: center; }
        .mindmap-section { margin-top: 12px; padding: 12px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e5e7eb; }
        .mindmap-label { font-size: 13px; font-weight: 600; color: #495057; margin-bottom: 8px; }
        .mindmap-mermaid { display: flex; justify-content: center; margin-bottom: 8px; overflow-x: auto; }
        .mindmap-mermaid .mermaid { display: flex; justify-content: center; }
        .mindmap-download { width: 100%; padding: 8px; border: 1px solid #dee2e6; border-radius: 6px; background: white; cursor: pointer; font-size: 13px; color: #495057; transition: background 0.2s; }
        .mindmap-download:hover { background: #e9ecef; }
        .loading { color: #9ca3af; font-style: italic; }
        .context-badge { display: inline-block; background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-top: 8px; }
        @media (max-width: 900px) { .mindmap-panel { display: none !important; } }
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
            <div class="input-area">
                <textarea id="userInput" placeholder="输入你的问题..." rows="1"></textarea>
                <button class="btn-send" id="sendBtn">➤</button>
            </div>
        </div>
        <div class="mindmap-panel" id="mindmapPanel">
            <div class="panel-header">📊 知识点思维导图</div>
            <div class="panel-body" id="mindmapBody"></div>
        </div>
    </div>
    <script>
        mermaid.initialize({ startOnLoad: true, theme: 'default' });
        let sessionId = localStorage.getItem('sessionId') || '';
        if (!sessionId) { sessionId = crypto.randomUUID(); localStorage.setItem('sessionId', sessionId); }

        const messagesEl = document.getElementById('messages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const mindmapPanel = document.getElementById('mindmapPanel');
        const mindmapBody = document.getElementById('mindmapBody');

        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
        });

        function addMessage(role, content, metadata = {}) {
            const msgEl = document.createElement('div');
            msgEl.className = `message ${role}`;
            let bubbleContent = content;
            // 思维导图渲染在气泡内，不显示源码
            if (metadata.mindmap && metadata.mindmap.trim()) {
                setTimeout(() => renderMindmapInline(msgEl, metadata.mindmap), 100);
            }
            // 显示上下文信息
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
            section.innerHTML = '<div class="mindmap-label">📊 知识点思维导图</div>';

            const mermaidDiv = document.createElement('div');
            mermaidDiv.className = 'mindmap-mermaid';
            mermaidDiv.innerHTML = `<div class="mermaid">${mermaidCode}</div>`;
            section.appendChild(mermaidDiv);

            // 下载按钮
            const downloadBtn = document.createElement('button');
            downloadBtn.className = 'mindmap-download';
            downloadBtn.textContent = '📥 下载思维导图';
            downloadBtn.addEventListener('click', () => downloadMindmap(mermaidCode));
            section.appendChild(downloadBtn);

            bubble.appendChild(section);

            // 渲染 mermaid
            try {
                mermaid.run({ nodes: mermaidDiv.querySelectorAll('.mermaid') });
            } catch (e) {
                mermaidDiv.innerHTML = `<pre>${mermaidCode}</pre>`;
            }
        }

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
                // 降级：下载 SVG 源码
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

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;
            addMessage('user', text);
            const loadingEl = document.createElement('div');
            loadingEl.className = 'message assistant';
            loadingEl.id = 'loading-msg';
            loadingEl.innerHTML = `<div class="avatar">🤖</div><div class="bubble loading">小智老师正在思考...</div>`;
            messagesEl.appendChild(loadingEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId, user_input: text})
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

        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        });
    </script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
