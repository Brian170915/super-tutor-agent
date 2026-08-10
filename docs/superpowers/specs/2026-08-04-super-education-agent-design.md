# 超级教育智能体系统设计文档

## Context

当前项目 `ai-super-tutor` 是一个极简的 LangChain + FastAPI 单链 Demo（`agent.py`），仅实现了一个简单的文本问答链。用户希望将其扩展为一个完整的初中教育智能体，支持多模态输入（OCR识别试卷）、RAG知识库问答、知识点思维导图等功能。用户在 `pythonProject` 中已有成熟的 RAG（ChromaDB + RRF双路检索）和 LangGraph 实践经验，本设计将复用这些已有能力。

## 需求摘要

- **目标学段**：初中
- **技术栈**：LangChain + LangGraph + Agnes API (OCR) + ChromaDB (RAG) + DashScope (Qwen LLM)
- **前端**：Vue 3 + FastAPI
- **MVP 范围**：答疑 + OCR识别试卷 + RAG知识库 + 知识点思维导图
- **后续迭代**：学习路径规划、智能出卷、考情分析
- **用户体系**：不需要（使用 session_id + localStorage 管理会话）

## 架构方案

### 推荐：单图多节点 LangGraph Agent（方案A）

使用 LangGraph `StateGraph` 构建单一状态图，包含多个职责单一的节点，通过条件边路由。

**核心优势：**
- 与现有 LangGraph 经验高度兼容
- 单一控制流，易于调试和 LangSmith 追踪
- 前端只需一个 `/chat` 端点
- 扩展新功能是增量操作（新增节点即可）

### 架构概览

```
FastAPI 后端 (port 8000)
├── POST /chat        → LangGraph Agent（主对话端点）
├── POST /ingest      → 知识库构建（上传教材/教辅）
├── GET  /knowledge-structure → 获取知识点体系
└── GET  /session/<id>    → 获取会话历史

LangGraph 流程图:
  START → QueryNode → RouteNode → (有图片) OCRNode → KnowledgeExtractNode → RAGNode
                                                              ↘ (无图片) ──────────→ RAGNode
                                                                                              ↓
                                                                                         ChatNode → ThoughtNode → END
```

## 项目结构

```
ai-super-tutor/
├── .env                          # 环境变量（已有）
├── agent.py                      # 入口（重构）
├── agent/                        # 核心代码目录
│   ├── __init__.py
│   ├── state.py                  # AgentState 定义
│   ├── graph.py                  # StateGraph 构建
│   ├── nodes.py                  # 各节点实现
│   ├── prompts.py                # Prompt 模板
│   ├── tools.py                  # 工具函数（OCR, RAG检索）
│   └── sessions.py              # 会话管理（无用户体系）
├── rag/                          # RAG 模块
│   ├── __init__.py
│   ├── pipeline.py              # RAG 处理管道（复用pythonProject）
│   ├── vectorstore.py           # Chroma 向量库
│   ├── bm25.py                  # BM25 检索
│   ├── ingestor.py              # 知识库增量同步
│   └── knowledge_structure.py   # 初中知识点体系
├── ocr/                          # OCR 模块
│   ├── __init__.py
│   └── agnes_client.py          # Agnes API 客户端
├── static/                       # 静态资源
│   └── js/
│       └── mermaid.min.js       # Mermaid 渲染库
├── templates/                    # HTML 模板
│   └── index.html               # 前端页面
└── requirements.txt             # 依赖声明
```

## 核心模块设计

### 1. State 定义 (`agent/state.py`)

```python
from typing import TypedDict, Optional, Annotated, List
import operator
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    # 输入
    user_input: str
    image_data: Optional[str]          # Base64 编码的图片
    
    # OCR 处理
    ocr_result: Optional[str]          # Agnes OCR 识别文本
    ocr_knowledge_points: List[str]    # LLM 提取的知识点
    
    # RAG 检索
    retrieved_docs: List[dict]         # 检索到的文档片段
    rephrased_query: str               # 重写后的查询词
    
    # 输出
    answer: str                        # LLM 回答
    mindmap_mermaid: str               # Mermaid 格式思维导图
    
    # 会话
    session_id: str
    knowledge_structure: dict          # 知识点体系
```

### 2. LangGraph 图构建 (`agent/graph.py`)

```python
from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import (
    query_node, route_node, ocr_node,
    rag_node, knowledge_extract_node,
    chat_node, thought_node
)

def build_graph():
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("query", query_node)
    graph.add_node("route", route_node)
    graph.add_node("ocr", ocr_node)
    graph.add_node("rag", rag_node)
    graph.add_node("knowledge_extract", knowledge_extract_node)
    graph.add_node("chat", chat_node)
    graph.add_node("thought", thought_node)
    
    # 添加边
    graph.add_edge(START, "query")
    graph.add_conditional_edges(
        "route",
        route_node,
        {"ocr": "ocr", "chat": "rag"}
    )
    graph.add_edge("ocr", "knowledge_extract")
    graph.add_edge("knowledge_extract", "rag")
    graph.add_edge("rag", "chat")
    graph.add_edge("chat", "thought")
    graph.add_edge("thought", END)
    
    return graph.compile()
```

### 3. 节点职责表

| 节点 | 职责 | 关键逻辑 |
|------|------|----------|
| `query_node` | 接收并解析输入 | 提取 user_input, image_data, session_id |
| `route_node` | 路由决策 | 有图片 → OCR路径，无图片 → 直接RAG |
| `ocr_node` | Agnes OCR | 调用 Agnes API，返回识别文本 |
| `knowledge_extract_node` | 知识点提取 | LLM 从 OCR 结果提取知识点列表 |
| `rag_node` | 检索知识库 | 复用 pythonProject 的 RRF 双路检索 |
| `chat_node` | 生成回答 | LLM + RAG 上下文生成答疑 |
| `thought_node` | 生成思维导图 | LLM 生成 Mermaid 格式知识点图谱 |

### 4. RAG 模块 (`rag/`)

**复用 pythonProject 的已有实现：**
- `pipeline.py` 中的 `RAGPipeline` 类（双路检索 + RRF 融合）
- `vectorstore.py` 中的 Chroma 管理
- `ingestor.py` 中的增量同步

**初中知识点体系 (`rag/knowledge_structure.py`)：**

```python
KNOWLEDGE_STRUCTURE = {
    "数学": {
        "七年级上": ["有理数", "整式加减", "一元一次方程", "几何图形初步"],
        "七年级下": ["相交线与平行线", "实数", "平面直角坐标系", "二元一次方程组"],
        "八年级上": ["三角形", "全等三角形", "轴对称", "整式的乘除"],
        "八年级下": ["特殊平行四边形", "反比例函数", "勾股定理", "数据分析"],
        "九年级": ["二次函数", "圆", "概率与统计", "相似三角形"]
    },
    "物理": {
        "八年级": ["声现象", "光现象", "透镜及其应用", "质量与密度"],
        "九年级": ["力学", "电学", "能量", "电磁学"]
    },
    "化学": {
        "九年级": ["物质变化", "空气与氧气", "水与氢气", "金属", "酸碱盐"]
    },
    "语文": {
        "七年级": ["古代诗歌", "记叙文", "散文", "文言文"],
        "八年级": ["说明文", "议论文", "文言文", "现代诗"],
        "九年级": ["中考古诗文", "议论文", "名著导读"]
    }
}
```

### 5. OCR 模块 (`ocr/agnes_client.py`)

```python
import base64
from langchain_openai import ChatOpenAI
import os

class AgnesOCRClient:
    """使用 OpenAI 兼容协议的 Agnes 视觉模型进行 OCR"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "agnes-vision"):
        # 复用 langchain-openai 的 ChatOpenAI 客户端
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0,
        )
    
    def recognize(self, image_base64: str) -> str:
        """
        调用 Agnes 视觉模型进行 OCR 识别
        
        Args:
            image_base64: Base64 编码的图片数据
            
        Returns:
            识别出的文本内容
        """
        prompt = """请识别图片中的所有文字，保持原文格式和排版。如果是试卷，请按照题目顺序输出。"""
        
        response = self.llm.invoke([
            ("system", "你是一个专业的OCR助手，擅长识别试卷、作业等文档中的文字。"),
            ("human", [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ])
        ])
        
        return response.content
```

**.env 配置：**
```bash
# 已有配置
DASHSCOPE_API_KEY=sk-xxx
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.7-max-2026-05-17

# Agnes 视觉模型配置（假设使用同一接口或独立接口）
AGNES_API_KEY=xxx
AGNES_BASE_URL=https://agnes-api.example.com/v1  # 或复用 DashScope
AGNES_MODEL=agnes-vision
```

### 6. 会话管理 (`agent/sessions.py`)

```python
import uuid
import time
from typing import Dict, List

# 内存存储（MVP）
_sessions: Dict[str, dict] = {}

def create_session() -> str:
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"messages": [], "session_start": time.time()}
    return session_id

def get_session(session_id: str) -> dict:
    return _sessions.get(session_id)

def add_message(session_id: str, role: str, content: str):
    _sessions[session_id]["messages"].append({
        "role": role, "content": content, "timestamp": time.time()
    })
```

## API 设计

```python
# agent.py (重构后)
from fastapi import FastAPI, UploadFile, File
from langserve import add_routes
from agent.graph import build_graph

app = FastAPI(title="超级教育智能体", version="2.0.0")

# 编译 LangGraph
graph = build_graph()

# 添加 LangServe 路由
add_routes(app, graph, path="/chat")

# 知识库构建端点
@app.post("/ingest")
async def ingest_knowledge(file: UploadFile = File(...)):
    # 调用 rag.ingestor 处理上传文件
    pass

# 知识点结构端点
@app.get("/knowledge-structure")
async def get_knowledge_structure():
    from rag.knowledge_structure import KNOWLEDGE_STRUCTURE
    return KNOWLEDGE_STRUCTURE

# 会话端点
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    from agent.sessions import get_session
    return get_session(session_id)
```

## 前端设计

**技术选型：** Vue 3 + Element Plus + Mermaid.js

**核心页面：**

1. **对话界面**
   - 聊天窗口（显示对话历史）
   - 输入框（支持文字输入）
   - 图片上传按钮（支持试卷/作业拍照）
   - 发送按钮

2. **知识点思维导图**
   - 使用 Mermaid.js 渲染
   - 支持缩放、平移
   - 可切换不同学科/年级

3. **知识库管理**（管理员）
   - 上传教材/教辅文件
   - 查看知识库状态
   - 触发增量同步

**交互流程：**

```
学生上传试卷图片
    │
    ▼
OCR 识别题目
    │
    ▼
显示识别结果（可编辑）
    │
    ▼
Agent 分析知识点
    │
    ▼
生成答疑回答 + 思维导图
    │
    ▼
学生提问（多轮对话）
```

## 依赖清单

```txt
# requirements.txt
# LLM & Agent
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-core>=0.2.0
langgraph>=0.1.0
langserve[all]>=0.2.0

# RAG
langchain-chroma>=0.1.0
langchain-community>=0.2.0
chromadb>=0.4.0
rank-bm25>=0.2.0

# Web
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
python-dotenv>=1.0.0

# 文档处理
pymupdf>=1.24.0
pillow>=10.0.0

# 工具
requests>=2.31.0
```

## 实施计划

### 阶段一：基础架构（1周）
1. 搭建项目目录结构
2. 定义 `AgentState` 状态类
3. 实现 `build_graph()` 图构建
4. 实现 `QueryNode` 和 `RouteNode`
5. 配置 FastAPI 路由

### 阶段二：核心功能（2周）
1. 集成 Agnes OCR 客户端
2. 复用 RAG Pipeline（从 pythonProject）
3. 实现 `OCRNode`、`RAGNode`、`ChatNode`
4. 实现知识点提取和思维导图生成

### 阶段三：前端对接（1周）
1. 搭建 Vue 前端项目
2. 实现对话界面
3. 集成 Mermaid 渲染
4. 对接后端 API

### 阶段四：测试优化（1周）
1. 单元测试各节点
2. 端到端集成测试
3. LangSmith tracing 验证
4. 性能优化（异步、流式输出）

## 后续迭代规划

### 第二阶段：学习路径规划
- 新增 `PathPlanNode`
- 基于学生错题数据分析薄弱知识点
- 生成个性化学习路径（Mermaid 流程图）

### 第三阶段：智能出卷
- 新增 `ExamGeneratorNode`
- 基于知识点难度和类型生成变式题
- 支持导出 PDF/Word

### 第四阶段：考情分析
- 新增 `AnalysisNode`
- OCR 识别多份试卷
- 统计知识点掌握情况
- 生成考情报告

## 关键文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `agent.py` | 重构 | 现有 Demo 入口，需重写 |
| `agent/state.py` | 新建 | AgentState 定义 |
| `agent/graph.py` | 新建 | LangGraph 图构建 |
| `agent/nodes.py` | 新建 | 各节点实现 |
| `agent/prompts.py` | 新建 | Prompt 模板 |
| `agent/tools.py` | 新建 | 工具函数 |
| `agent/sessions.py` | 新建 | 会话管理 |
| `rag/pipeline.py` | 复用 | 从 pythonProject 复制 |
| `rag/vectorstore.py` | 复用 | 从 pythonProject 复制 |
| `rag/ingestor.py` | 复用 | 从 pythonProject 复制 |
| `rag/knowledge_structure.py` | 新建 | 初中知识点体系 |
| `ocr/agnes_client.py` | 新建 | Agnes OCR 客户端 |

## 验证方案

1. **单元测试**：每个节点独立测试
   - `test_query_node.py`
   - `test_ocr_node.py`
   - `test_rag_node.py`
   - `test_chat_node.py`

2. **集成测试**：端到端流程
   - 文本问答流程
   - 图片 OCR + 答疑流程
   - 思维导图生成流程

3. **LangSmith 验证**
   - 启用 `LANGCHAIN_TRACING_V2=true`
   - 查看每个节点的执行时间和token消耗
   - 验证状态流转正确性

## 注意事项

1. **Agnes API**：使用 OpenAI 兼容协议，可通过 `langchain_openai.ChatOpenAI` 直接调用，配置 `base_url` 和 `api_key` 即可
2. **会话持久化**：MVP 使用内存存储，生产环境需迁移到 Redis
3. **知识点体系**：需要根据实际初中教材版本调整
4. **前端技术栈**：用户选择了 Vue，但需要确认是否有具体的 UI 框架偏好

---

*文档创建时间：2026-08-04*
*版本：v1.0*
