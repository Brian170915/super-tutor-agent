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
        .input-area { padding: 16px 20px; background: white; border-top: 1px solid #e5e7eb; display: flex; gap: 12px; }
        .input-area textarea { flex: 1; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; resize: none; font-size: 14px; min-height: 44px; max-height: 120px; font-family: inherit; }
        .input-area textarea:focus { outline: none; border-color: #667eea; }
        .input-area button { width: 44px; height: 44px; border: none; border-radius: 12px; cursor: pointer; font-size: 20px; transition: background 0.2s; }
        .btn-send { background: #667eea; color: white; }
        .btn-send:hover { background: #5a6fd6; }
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
            <div class="input-area">
                <textarea id="userInput" placeholder="输入你的问题..." rows="1"></textarea>
                <button class="btn-send" id="sendBtn">➤</button>
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
        const zoomOverlay = document.getElementById('zoomOverlay');
        const zoomWrap = document.getElementById('zoomWrap');
        const zoomContent = document.getElementById('zoomContent');

        // 缩放状态
        let zoomLevel = 1;
        let currentMermaidCode = '';

        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
        });

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

            // 点击整块区域打开缩放
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

        // ============ 缩放功能 ============
        function openZoom(mermaidCode) {
            currentMermaidCode = mermaidCode;
            zoomLevel = 1;
            zoomContent.innerHTML = `<div class="mermaid">${mermaidCode}</div>`;
            zoomOverlay.classList.add('active');
            zoomWrap.style.transform = '';
            zoomWrap.style.transformOrigin = 'top left';

            // 渲染 mermaid
            try {
                mermaid.run({ nodes: zoomContent.querySelectorAll('.mermaid') });
            } catch (e) {}

            // 等 SVG 生成后再缩放
            setTimeout(() => applyZoom(), 100);
        }

        function closeZoom() {
            zoomOverlay.classList.remove('active');
        }

        function applyZoom() {
            const svg = zoomContent.querySelector('svg');
            if (!svg) return;
            // 获取 SVG 原始尺寸，放大显示
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
            if (svg) {
                svg.style.transform = `scale(${zoomLevel})`;
            }
        }

        function zoomOut() {
            zoomLevel = Math.max(zoomLevel / 1.3, 0.3);
            const svg = zoomContent.querySelector('svg');
            if (svg) {
                svg.style.transform = `scale(${zoomLevel})`;
            }
        }

        function zoomReset() {
            zoomLevel = 1;
            const svg = zoomContent.querySelector('svg');
            if (svg) svg.style.transform = 'scale(1)';
        }

        // 鼠标滚轮缩放
        zoomWrap.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (e.deltaY < 0) zoomIn();
            else zoomOut();
        }, { passive: false });

        // 点击遮罩关闭
        zoomOverlay.addEventListener('click', (e) => {
            if (e.target === zoomOverlay) closeZoom();
        });

        // ESC 关闭
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeZoom();
        });

        // 拖拽平移
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
