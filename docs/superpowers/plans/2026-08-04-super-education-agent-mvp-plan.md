# 超级教育智能体 MVP 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有的极简 LangChain 单链 Demo 重构为基于 LangGraph 的多节点教育智能体，支持文本答疑、OCR 试卷识别、RAG 知识库问答、知识点思维导图生成。

**Architecture:** 使用 LangGraph StateGraph 构建单一状态图，包含 QueryNode → RouteNode → OCRNode/RAGNode → KnowledgeExtractNode → ChatNode → ThoughtNode。RAG 模块复用 pythonProject 的 ChromaDB + BM25 双路检索实现。OCR 使用 Agnes 视觉模型（OpenAI 兼容协议）。

**Tech Stack:** LangChain 1.2, LangGraph 1.1, langchain-openai, langchain-chroma, langchain-community (BM25Retriever), FastAPI, python-dotenv, DashScope/Qwen LLM

## Global Constraints

- LLM: DashScope API (base_url: https://dashscope.aliyuncs.com/compatible-mode/v1, model: qwen3.7-max-2026-05-17)
- LangChain version: >= 0.2.0 (use langchain_classic, langchain_core, langchain_openai, langchain_chroma, langchain_community)
- LangGraph version: >= 0.1.0
- Agnes OCR: OpenAI compatible protocol via langchain_openai.ChatOpenAI
- Session management: in-memory dict with UUID session_id (no user auth)
- Knowledge structure: Mermaid graph TD format for middle school (初中) knowledge points
- All code must be in Chinese project context but English variable names

---

### Task 1: 项目初始化与依赖配置

**Files:**
- Create: `requirements.txt`
- Modify: `.env` (add AGNES config)
- Create: `agent/__init__.py`
- Create: `rag/__init__.py`
- Create: `ocr/__init__.py`

**Interfaces:**
- Consumes: none (setup task)
- Produces: `requirements.txt` with all dependencies, `.env` with all API keys

- [ ] **Step 1: 创建 requirements.txt**

```txt
# LLM & Agent Framework
langchain==1.2.17
langchain-classic==1.0.4
langchain-core==1.3.2
langchain-openai==1.2.1
langchain-chroma==1.1.0
langchain-community==0.4.1
langchain-text-splitters==1.1.2

# LangGraph
langgraph==1.1.10

# LangServe
langserve[all]==0.3.0
pydantic==2.13.3

# Web
fastapi==0.136.3
uvicorn[standard]==0.46.0
python-multipart==0.0.26

# RAG & Vector Store
chromadb==1.5.8
rank-bm25>=0.2.0

# OCR & Document Processing
pymupdf>=1.24.0
pillow>=10.0.0

# Utilities
python-dotenv==1.2.2
requests>=2.31.0
```

- [ ] **Step 2: 更新 .env 添加 Agnes 配置**

读取现有 `.env` 文件，在其末尾追加：
```
# Agnes 视觉模型（OCR 使用，OpenAI 兼容协议）
AGNES_API_KEY=${DASHSCOPE_API_KEY}
AGNES_BASE_URL=${BASE_URL}
AGNES_MODEL=qwen-vl-max
```

- [ ] **Step 3: 创建包初始化文件**

创建以下空 `__init__.py` 文件：
- `agent/__init__.py`
- `rag/__init__.py`
- `ocr/__init__.py`

- [ ] **Step 4: 安装依赖并验证**

Run: `pip install -r requirements.txt`

验证关键包可导入：
```python
import langchain
import langgraph
import langchain_openai
import langchain_chroma
import fastapi
print("All imports OK")
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .env agent/__init__.py rag/__init__.py ocr/__init__.py
git commit -m "chore: add requirements.txt and package init files"
```

---

### Task 2: 知识点体系与 RAG 基础设施

**Files:**
- Create: `rag/knowledge_structure.py`
- Create: `rag/vectorstore.py`
- Create: `rag/bm25_index.py`
- Create: `rag/pipeline.py`
- Create: `rag/ingestor.py`

**Interfaces:**
- Consumes: none (foundation task)
- Produces: `RAGInfrastructure` class with `vectorstore`, `bm25_retriever`, `llm`, `pipeline` properties

- [ ] **Step 1: 创建知识点体系 `rag/knowledge_structure.py`**

```python
"""
初中知识点体系定义
用于思维导图生成和知识点导航
"""

KNOWLEDGE_STRUCTURE: dict = {
    "数学": {
        "七年级上": ["有理数", "整式加减", "一元一次方程", "几何图形初步"],
        "七年级下": ["相交线与平行线", "实数", "平面直角坐标系", "二元一次方程组"],
        "八年级上": ["三角形", "全等三角形", "轴对称", "整式的乘除与因式分解"],
        "八年级下": ["特殊平行四边形", "反比例函数", "勾股定理", "数据分析与概率初步"],
        "九年级": ["二次函数", "圆", "相似三角形", "锐角三角函数", "投影与视图"]
    },
    "物理": {
        "八年级": ["声现象", "光现象", "透镜及其应用", "质量与密度", "力学初步"],
        "九年级": ["电学", "电与磁", "能量与能源", "声学综合"]
    },
    "化学": {
        "九年级": ["化学变化与性质", "空气与氧气", "水与氢气", "碳和碳的氧化物",
                   "金属与金属材料", "酸碱盐", "化学与生活"]
    },
    "语文": {
        "七年级": ["古代诗歌阅读", "记叙文阅读", "散文阅读", "文言文入门"],
        "八年级": ["说明文阅读", "议论文阅读", "文言文进阶", "现代诗鉴赏"],
        "九年级": ["中考古诗文背诵", "议论文专题", "名著导读", "写作技巧"]
    },
    "英语": {
        "七年级": ["基本句型", "一般现在时", "名词与代词", "基础词汇"],
        "八年级": ["一般过去时", "比较级与最高级", "定语从句入门", "动词短语"],
        "九年级": ["现在完成时", "宾语从句", "被动语态", "中考词汇"]
    }
}


def get_subjects() -> list[str]:
    """返回所有学科列表"""
    return list(KNOWLEDGE_STRUCTURE.keys())


def get_grades(subject: str) -> list[str]:
    """返回指定学科的所有年级"""
    return list(KNOWLEDGE_STRUCTURE.get(subject, {}).keys())


def get_topics(subject: str, grade: str) -> list[str]:
    """返回指定学科+年级的所有知识点"""
    return KNOWLEDGE_STRUCTURE.get(subject, {}).get(grade, [])
```

- [ ] **Step 2: 创建向量库管理器 `rag/vectorstore.py`**

```python
"""
Chroma 向量库管理器 - 基于 pythonProject 的 VectorStoreManager 适配
"""
import os
import hashlib
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "education_kb",
        embedding_model: str = "BAAI/bge-small-zh-v1.5"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    
    def add_documents(self, chunks: List[Document]) -> int:
        """添加文档，自动去重，返回新增数量"""
        if not chunks:
            return 0
        
        # MD5 去重
        all_ids = [hashlib.md5(chunk.page_content.encode()).hexdigest() for chunk in chunks]
        existing_data = self.vector_store.get(ids=all_ids)
        existing_ids = set(existing_data['ids'])
        
        new_chunks = []
        new_ids = []
        for i, chunk_id in enumerate(all_ids):
            if chunk_id not in existing_ids:
                new_chunks.append(chunks[i])
                new_ids.append(chunk_id)
        
        if new_chunks:
            self.vector_store.add_documents(new_chunks, ids=new_ids)
        
        return len(new_chunks)
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """向量相似度搜索"""
        return self.vector_store.similarity_search(query, k=k)
    
    def persist(self):
        """持久化到磁盘"""
        self.vector_store.persist()
    
    def load_local(self):
        """从磁盘加载（Chroma 自动加载，此方法保留接口一致性）"""
        pass
```

- [ ] **Step 3: 创建 BM25 索引 `rag/bm25_index.py`**

```python
"""
BM25 关键词检索器 - 基于 langchain_community.BM25Retriever
"""
import os
import pickle
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class BM25Index:
    CACHE_PATH = "./bm25_retriever.pkl"
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.retriever: BM25Retriever = None
    
    def build(self, documents: List[Document]):
        """从文档构建 BM25 索引"""
        chunks = self.splitter.split_documents(documents)
        self.retriever = BM25Retriever.from_documents(chunks)
        self.retriever.k = 4
        self._save_cache()
        return chunks
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """搜索"""
        if self.retriever is None:
            return []
        return self.retriever.invoke(query)
    
    def _save_cache(self):
        """缓存检索器"""
        if self.retriever:
            with open(self.CACHE_PATH, 'wb') as f:
                pickle.dump(self.retriever, f)
    
    def load_cache(self):
        """从缓存加载"""
        if os.path.exists(self.CACHE_PATH):
            with open(self.CACHE_PATH, 'rb') as f:
                self.retriever = pickle.load(f)
            return True
        return False
```

- [ ] **Step 4: 创建 RAG Pipeline `rag/pipeline.py`**

```python
"""
RAG 处理管道 - 双路检索 + RRF 融合
基于 pythonProject/core/pipeline.py 适配
"""
import json
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel


def reciprocal_rank_fusion(results: list[list], k: int = 30) -> List[Document]:
    """RRF 融合算法"""
    fused_scores = {}
    for doc_list in results:
        for rank, doc in enumerate(doc_list):
            if not doc:
                continue
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
            doc_str = json.dumps(doc_dict, sort_keys=True, ensure_ascii=False)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0.0
            fused_scores[doc_str] += 1.0 / (rank + k)
    
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    reranked = []
    for doc_str, _ in sorted_docs:
        data = json.loads(doc_str)
        reranked.append(Document(page_content=data["page_content"], metadata=data["metadata"]))
    return reranked


class RAGPipeline:
    def __init__(
        self,
        llm: BaseLanguageModel,
        vs_manager,
        bm25_index: 'BM25Index',
        rephrase_prompt: ChatPromptTemplate,
        response_prompt: ChatPromptTemplate
    ):
        self.llm = llm
        self.vs_manager = vs_manager
        self.bm25_index = bm25_index
        self.rephrase_prompt = rephrase_prompt
        self.response_prompt = response_prompt
    
    def retrieve(self, query: str) -> List[str]:
        """双路检索 + RRF 融合"""
        # 1. Query 重写
        chain = self.rephrase_prompt | self.llm
        rephrased = chain.invoke({"question": query}).content.strip()
        
        # 2. 向量检索
        doc_list_vector = self.vs_manager.search(rephrased, k=4)
        
        # 3. BM25 检索
        doc_list_bm25 = self.bm25_index.search(rephrased, k=4)
        
        # 4. RRF 融合
        fused = reciprocal_rank_fusion([doc_list_vector, doc_list_bm25], k=30)
        
        # 5. 取 Top 2
        return [doc.page_content for doc in fused[:2]]
    
    def generate(self, query: str, context_docs: List[str]) -> str:
        """生成回答"""
        context = "\n\n".join(context_docs)
        formatted = self.response_prompt.format(question=query, context=context)
        return self.llm.invoke(formatted).content
    
    def query(self, query: str) -> str:
        """完整 RAG 查询"""
        context = self.retrieve(query)
        return self.generate(query, context)
```

- [ ] **Step 5: 创建知识库增量同步器 `rag/ingestor.py`**

```python
"""
知识库增量同步器
"""
import os
import json
import hashlib
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KBIngestor:
    def __init__(
        self,
        data_folder: str,
        vs_manager,
        bm25_index,
        manifest_path: str = "./ingest_manifest.json"
    ):
        self.data_folder = data_folder
        self.vs_manager = vs_manager
        self.bm25_index = bm25_index
        self.manifest_path = manifest_path
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    @staticmethod
    def _md5(file_path: str) -> str:
        h = hashlib.md5()
        with open(file_path, 'rb') as f:
            h.update(f.read())
        return h.hexdigest()
    
    def _load_manifest(self) -> dict:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_manifest(self, manifest: dict):
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=4)
    
    def sync(self) -> int:
        """增量同步，返回新增文档数"""
        manifest = self._load_manifest()
        
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
            return 0
        
        current_files = [
            f for f in os.listdir(self.data_folder)
            if os.path.isfile(os.path.join(self.data_folder, f))
        ]
        
        added_count = 0
        for filename in current_files:
            file_path = os.path.join(self.data_folder, filename)
            current_md5 = self._md5(file_path)
            
            if manifest.get(filename) == current_md5:
                continue
            
            # 加载文档
            docs = self._load_file(file_path, filename)
            chunks = self.splitter.split_documents(docs)
            
            if chunks:
                self.vs_manager.add_documents(chunks)
                added_count += len(chunks)
            
            manifest[filename] = current_md5
        
        # 清理已删除文件
        deleted = [f for f in manifest if f not in current_files]
        for fn in deleted:
            del manifest[fn]
        
        self._save_manifest(manifest)
        self.bm25_index._save_cache()
        return added_count
    
    def _load_file(self, file_path: str, filename: str) -> List[Document]:
        """根据文件类型加载文档"""
        from langchain_community.document_loaders import (
            PyMuPDFLoader, TextLoader, PythonLoader
        )
        
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.pdf':
            loader = PyMuPDFLoader(file_path)
        elif ext in ('.txt', '.md'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == '.py':
            loader = PythonLoader(file_path)
        else:
            return []
        
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_file'] = filename
        return docs
```

- [ ] **Step 6: Commit**

```bash
git add rag/
git commit -m "feat: add RAG infrastructure (vectorstore, bm25, pipeline, ingestor, knowledge structure)"
```

---

### Task 3: OCR 模块

**Files:**
- Create: `ocr/agnes_client.py`

**Interfaces:**
- Consumes: `.env` AGNES_API_KEY, AGNES_BASE_URL, AGNES_MODEL
- Produces: `AgnesOCRClient.recognize(image_base64: str) -> str`

- [ ] **Step 1: 创建 Agnes OCR 客户端 `ocr/agnes_client.py`**

```python
"""
Agnes 视觉模型 OCR 客户端
使用 OpenAI 兼容协议调用多模态模型
"""
import os
import base64
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class AgnesOCRClient:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("AGNES_API_KEY", os.getenv("DASHSCOPE_API_KEY")),
            base_url=os.getenv("AGNES_BASE_URL", os.getenv("BASE_URL")),
            model=os.getenv("AGNES_MODEL", "qwen-vl-max"),
            temperature=0,
        )
    
    def _image_to_base64(self, image_bytes: bytes) -> str:
        """将图片字节转为 base64"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def recognize(self, image_input) -> str:
        """
        OCR 识别图片中的文字
        
        Args:
            image_input: bytes 或 PIL.Image 或文件路径
            
        Returns:
            识别出的文本内容
        """
        if isinstance(image_input, str):
            # 文件路径
            with open(image_input, 'rb') as f:
                image_bytes = f.read()
        elif hasattr(image_input, 'read'):
            # 文件对象
            image_bytes = image_input.read()
        else:
            # 已经是 bytes
            image_bytes = image_input
        
        b64 = self._image_to_base64(image_bytes)
        
        prompt = """请识别图片中的所有文字，保持原文格式和排版。如果是试卷，请按照题目顺序逐题输出，每道题标注题号和题目内容。"""
        
        response = self.llm.invoke([
            ("system", "你是一个专业的OCR助手，擅长识别试卷、作业等教育文档中的文字。请准确识别所有文字内容，包括题目、选项、公式等。"),
            ("human", [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ])
        ])
        
        return response.content
```

- [ ] **Step 2: Commit**

```bash
git add ocr/
git commit -m "feat: add Agnes OCR client module"
```

---

### Task 4: Agent State 与 Prompt 模板

**Files:**
- Create: `agent/state.py`
- Create: `agent/prompts.py`

**Interfaces:**
- Consumes: none
- Produces: `AgentState` TypedDict, prompt templates

- [ ] **Step 1: 创建 AgentState `agent/state.py`**

```python
"""
Agent 状态定义
"""
from typing import TypedDict, Optional, List, Annotated
import operator
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    # 输入
    user_input: str
    image_data: Optional[str]  # Base64 编码的图片
    
    # OCR 处理
    ocr_result: Optional[str]
    ocr_knowledge_points: List[str]
    
    # RAG 检索
    retrieved_docs: List[str]
    rephrased_query: str
    
    # 输出
    answer: str
    mindmap_mermaid: str
    
    # 会话
    session_id: str
```

- [ ] **Step 2: 创建 Prompt 模板 `agent/prompts.py`**

```python
"""
Prompt 模板定义
"""
from langchain_core.prompts import ChatPromptTemplate

# 查询重写 - 将学生问题转化为适合检索的关键词
REPHRASE_PROMPT = ChatPromptTemplate.from_template("""你是一位经验丰富的初中教师。
请分析学生的中文问题，提取出最适合在初中教材中检索的【核心知识点关键词】。

【严格要求】：
1. 只需输出优化后的中文检索关键词，关键词之间用空格隔开
2. 将口语化表达转化为教材标准术语
3. 绝对不要包含任何解释、标点或多余内容

学生问题: {question}
优化搜索词:""")

# 知识点提取 - 从 OCR 结果中提取知识点
EXTRACT_KNOWLEDGE_PROMPT = ChatPromptTemplate.from_template("""你是一位初中教师，请分析以下试卷/作业内容，提取出涉及的所有知识点。

【要求】：
1. 只输出知识点名称，每行一个
2. 使用初中教材的标准知识点名称
3. 最多提取 10 个知识点
4. 不要输出任何解释

试卷/作业内容:
{ocr_text}

知识点列表:""")

# 答疑回答 - 结合 RAG 上下文生成回答
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个友好的初中教育智能体，名叫"小智老师"。

你的职责：
1. 根据知识库中的信息解答学生的 questions
2. 回答要通俗易懂，适合初中生理解
3. 如果知识库中没有相关信息，诚实地告知学生
4. 鼓励性的语言，帮助学生建立学习信心

回答格式：
- 先给出清晰的结论
- 然后详细解释
- 如有必要，给出例题或记忆技巧"""),
    ("human", """知识库参考内容：
{context}

学生问题: {question}

请给出解答：""")
])

# 思维导图生成
MINDMAP_PROMPT = ChatPromptTemplate.from_template("""你是一位初中教育专家，请根据以下知识点生成 Mermaid 格式的思维导图。

【要求】：
1. 使用 graph TD 语法
2. 最多 3 层层级
3. 中心节点是主题
4. 使用中文标注
5. 知识点之间用逻辑关系连接

知识点: {knowledge_points}

Mermaid 格式:""")

# 试卷分析
EXAM_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位初中教师，请分析学生上传的试卷内容。

分析维度：
1. 试卷包含哪些知识点
2. 各知识点的题目数量和难度分布
3. 学生可能薄弱的内容
4. 学习建议

请给出结构化的分析报告。"""),
    ("human", """试卷内容:
{ocr_text}

请分析：""")
])
```

- [ ] **Step 3: Commit**

```bash
git add agent/state.py agent/prompts.py
git commit -m "feat: add AgentState and prompt templates"
```

---

### Task 5: Agent 节点实现

**Files:**
- Create: `agent/nodes.py`

**Interfaces:**
- Consumes: `AgentState`, AgnesOCRClient, RAGPipeline, LLM
- Produces: state updates for graph nodes

- [ ] **Step 1: 创建节点实现 `agent/nodes.py`**

```python
"""
LangGraph 节点实现
"""
import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agent.state import AgentState
from agent.prompts import (
    REPHRASE_PROMPT, EXTRACT_KNOWLEDGE_PROMPT,
    CHAT_PROMPT, MINDMAP_PROMPT
)
from ocr.agnes_client import AgnesOCRClient
from rag.pipeline import RAGPipeline
from rag.knowledge_structure import KNOWLEDGE_STRUCTURE

load_dotenv()

# 全局 LLM 实例
_llm = None
_ocr_client = None
_rag_pipeline = None
_bm25_index = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("BASE_URL"),
            model=os.getenv("LLM_MODEL"),
            temperature=0.1,
        )
    return _llm


def get_ocr_client() -> AgnesOCRClient:
    global _ocr_client
    if _ocr_client is None:
        _ocr_client = AgnesOCRClient()
    return _ocr_client


def get_rag_pipeline(vs_manager, bm25_index):
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(
            llm=get_llm(),
            vs_manager=vs_manager,
            bm25_index=bm25_index,
            rephrase_prompt=REPHRASE_PROMPT,
            response_prompt=CHAT_PROMPT
        )
    return _rag_pipeline


def query_node(state: AgentState) -> AgentState:
    """接收并初始化输入"""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    user_input = ""
    image_data = None
    session_id = state.get("session_id", "")
    
    if last_message and hasattr(last_message, 'content'):
        user_input = last_message.content
    
    # 从额外字段提取图片（LangGraph MessagesState 的 additional_kwargs）
    if last_message and hasattr(last_message, 'additional_kwargs'):
        extra = last_message.additional_kwargs
        if isinstance(extra, dict):
            image_data = extra.get("image_data")
            session_id = extra.get("session_id", session_id)
    
    return {
        "user_input": user_input,
        "image_data": image_data,
        "session_id": session_id,
        "ocr_result": None,
        "ocr_knowledge_points": [],
        "retrieved_docs": [],
        "rephrased_query": "",
        "answer": "",
        "mindmap_mermaid": "",
    }


def route_node(state: AgentState) -> Literal["ocr", "chat"]:
    """根据是否有图片路由"""
    if state.get("image_data"):
        return "ocr"
    return "chat"


def ocr_node(state: AgentState) -> AgentState:
    """调用 Agnes OCR 识别图片"""
    if not state.get("image_data"):
        return state
    
    image_b64 = state["image_data"]
    # 移除 data URI 前缀
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]
    
    import base64
    image_bytes = base64.b64decode(image_b64)
    
    ocr_client = get_ocr_client()
    ocr_text = ocr_client.recognize(image_bytes)
    
    return {
        **state,
        "ocr_result": ocr_text,
    }


def knowledge_extract_node(state: AgentState) -> AgentState:
    """从 OCR 结果提取知识点"""
    ocr_text = state.get("ocr_result", "")
    if not ocr_text:
        return state
    
    llm = get_llm()
    prompt = EXTRACT_KNOWLEDGE_PROMPT.format(ocr_text=ocr_text)
    response = llm.invoke(prompt).content
    
    # 解析知识点列表
    points = [p.strip() for p in response.strip().split('\n') if p.strip()]
    # 限制最多 10 个
    points = points[:10]
    
    return {
        **state,
        "ocr_knowledge_points": points,
    }


def rag_node(state: AgentState) -> AgentState:
    """检索 RAG 知识库"""
    llm = get_llm()
    
    # 构建检索查询
    query = state.get("user_input", "")
    ocr_text = state.get("ocr_result", "")
    
    # 如果有 OCR 结果，合并 OCR 文本作为上下文
    if ocr_text:
        # 先尝试从 OCR 文本中理解问题意图
        intent_prompt = f"""请分析以下试卷/作业内容，提取其中的学生可能提出的问题：
{ocr_text[:2000]}

如果内容中包含明确的题目，请概括题目大意。如果只是知识点列表，请输出"需要学生进一步提问"。
只输出概括，不要解释。"""
        intent = llm.invoke(intent_prompt).content
        if "需要学生进一步提问" not in intent:
            query = intent
    
    # 构建 RAG pipeline
    from rag.vectorstore import VectorStoreManager
    from rag.bm25_index import BM25Index
    
    vs_manager = VectorStoreManager()
    bm25_index = BM25Index()
    
    if not bm25_index.load_cache():
        # 尝试从 data 目录加载
        data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        if os.path.exists(data_folder):
            from rag.ingestor import KBIngestor
            ingestor = KBIngestor(data_folder, vs_manager, bm25_index)
            ingestor.sync()
    
    pipeline = get_rag_pipeline(vs_manager, bm25_index)
    
    # 检索
    retrieved_docs = pipeline.retrieve(query)
    
    return {
        **state,
        "retrieved_docs": retrieved_docs,
    }


def chat_node(state: AgentState) -> AgentState:
    """生成答疑回答"""
    llm = get_llm()
    
    query = state.get("user_input", "")
    ocr_text = state.get("ocr_result", "")
    retrieved_docs = state.get("retrieved_docs", [])
    
    # 构建上下文
    context_parts = []
    
    # 如果有 OCR 文本，将其作为额外上下文
    if ocr_text:
        context_parts.append(f"【试卷/作业内容】\n{ocr_text[:3000]}")
    
    # 如果有 RAG 检索结果
    if retrieved_docs:
        context_parts.append("【知识库参考】\n" + "\n\n".join(retrieved_docs))
    
    context = "\n\n".join(context_parts) if context_parts else "无额外上下文，请依靠模型自身知识解答。"
    
    # 构建消息
    messages = [
        ("system", """你是一个友好的初中教育智能体，名叫"小智老师"。

你的职责：
1. 根据提供的参考资料解答学生的 questions
2. 回答要通俗易懂，适合初中生理解
3. 如果参考资料中没有相关信息，诚实地告知学生
4. 鼓励性的语言，帮助学生建立学习信心
5. 对于试卷题目，先分析考点，再给出解题思路和答案"""),
        ("human", f"""参考资料：
{context}

学生问题: {query}

请给出解答：""")
    ]
    
    response = llm.invoke(messages)
    
    return {
        **state,
        "answer": response.content,
    }


def thought_node(state: AgentState) -> AgentState:
    """生成知识点思维导图"""
    llm = get_llm()
    
    knowledge_points = state.get("ocr_knowledge_points", [])
    user_input = state.get("user_input", "")
    
    # 尝试从用户问题或 OCR 结果中推断学科
    subject_hint = ""
    if knowledge_points:
        subject_hint = "已知知识点: " + ", ".join(knowledge_points[:5])
    
    prompt = MINDMAP_PROMPT.format(
        knowledge_points=state.get("ocr_knowledge_points", []) or user_input,
        subject_hint=subject_hint
    )
    
    response = llm.invoke(prompt)
    mermaid_code = response.content
    
    # 清理可能的前后标记
    for prefix in ["```mermaid", "```", "mermaid"]:
        if mermaid_code.startswith(prefix):
            mermaid_code = mermaid_code[len(prefix):].lstrip()
        if mermaid_code.endswith("```"):
            mermaid_code = mermaid_code[:-3].rstrip()
    
    return {
        **state,
        "mindmap_mermaid": mermaid_code,
    }
```

- [ ] **Step 2: Commit**

```bash
git add agent/nodes.py
git commit -m "feat: add all agent nodes (query, route, ocr, knowledge_extract, rag, chat, thought)"
```

---

### Task 6: LangGraph 图构建

**Files:**
- Create: `agent/graph.py`

**Interfaces:**
- Consumes: `agent.nodes` functions
- Produces: `build_graph() -> CompiledGraph`

- [ ] **Step 1: 创建图构建 `agent/graph.py`**

```python
"""
LangGraph 状态图构建
"""
from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import (
    query_node, route_node, ocr_node,
    knowledge_extract_node, rag_node,
    chat_node, thought_node
)


def build_graph():
    """构建并编译 Agent 状态图"""
    graph = StateGraph(AgentState)
    
    # 注册节点
    graph.add_node("query", query_node)
    graph.add_node("route", route_node)
    graph.add_node("ocr", ocr_node)
    graph.add_node("knowledge_extract", knowledge_extract_node)
    graph.add_node("rag", rag_node)
    graph.add_node("chat", chat_node)
    graph.add_node("thought", thought_node)
    
    # 添加边
    graph.add_edge(START, "query")
    
    # 条件路由：有图片走 OCR 路径，否则直接 RAG
    graph.add_conditional_edges(
        "query",
        route_node,
        {
            "ocr": "ocr",
            "chat": "rag"
        }
    )
    
    # OCR 路径
    graph.add_edge("ocr", "knowledge_extract")
    graph.add_edge("knowledge_extract", "rag")
    
    # 共同路径
    graph.add_edge("rag", "chat")
    graph.add_edge("chat", "thought")
    graph.add_edge("thought", END)
    
    # 编译
    app = graph.compile()
    return app
```

- [ ] **Step 2: Commit**

```bash
git add agent/graph.py
git commit -m "feat: add LangGraph state graph construction"
```

---

### Task 7: 主入口与 API 路由

**Files:**
- Modify: `agent.py`
- Create: `agent/sessions.py`

**Interfaces:**
- Consumes: `agent.graph.build_graph()`
- Produces: FastAPI app with `/chat`, `/ingest`, `/knowledge-structure` endpoints

- [ ] **Step 1: 创建会话管理 `agent/sessions.py`**

```python
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
```

- [ ] **Step 2: 重构主入口 `agent.py`**

```python
"""
超级教育智能体 - 主入口
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langserve import add_routes

from agent.graph import build_graph
from agent.sessions import create_session, add_message, get_messages
from rag.knowledge_structure import KNOWLEDGE_STRUCTURE, get_subjects, get_grades, get_topics
from rag.ingestor import KBIngestor
from rag.vectorstore import VectorStoreManager
from rag.bm25_index import BM25Index

load_dotenv()

app = FastAPI(
    title="超级教育智能体",
    description="基于 LangGraph 的初中教育智能体 - 答疑 + OCR + RAG + 思维导图",
    version="2.0.0"
)

# 编译 LangGraph
graph = build_graph()

# 添加 LangServe 路由
add_routes(app, graph, path="/chat")

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 会话存储
session_store = {}


@app.on_event("startup")
async def startup():
    """启动时初始化 RAG 基础设施"""
    from rag.vectorstore import VectorStoreManager
    from rag.bm25_index import BM25Index
    
    # 尝试加载已有的向量库
    vs_manager = VectorStoreManager()
    bm25_index = BM25Index()
    
    if not bm25_index.load_cache():
        data_folder = os.path.join(os.path.dirname(__file__), "data")
        if os.path.exists(data_folder):
            ingestor = KBIngestor(data_folder, vs_manager, bm25_index)
            ingestor.sync()
            print(f"知识库初始化完成")
        else:
            print("警告：data 目录不存在，RAG 检索可能返回空结果")
    else:
        print("BM25 缓存加载成功")


@app.post("/chat")
async def chat_endpoint(
    session_id: str = "",
    user_input: str = "",
    image: UploadFile = File(None)
):
    """
    主对话端点
    - 文本对话：直接问答
    - 图片上传：OCR 识别 → 知识点提取 → RAG 检索 → 答疑 → 思维导图
    """
    # 确保有 session
    if not session_id or session_id not in session_store:
        session_id = create_session()
        session_store[session_id] = {"messages": [], "session_start": __import__('time').time()}
    
    # 处理图片
    image_b64 = None
    if image and image.filename:
        image_bytes = await image.read()
        import base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        image_b64 = f"data:image/{image.filename.split('.')[-1] if '.' in image.filename else 'jpeg'};base64,{image_b64}"
    
    # 构建输入
    from langchain_core.messages import HumanMessage
    
    extra_kwargs = {}
    if image_b64:
        extra_kwargs["image_data"] = image_b64
    extra_kwargs["session_id"] = session_id
    
    human_msg = HumanMessage(
        content=user_input or "请分析这张试卷/图片",
        additional_kwargs=extra_kwargs
    )
    
    # 运行 graph
    result = graph.invoke({"messages": [human_msg]})
    
    # 记录消息
    answer = result.get("answer", "")
    add_message(session_id, "assistant", answer, {
        "mindmap": result.get("mindmap_mermaid", ""),
        "ocr_result": result.get("ocr_result", ""),
    })
    
    return {
        "session_id": session_id,
        "answer": answer,
        "mindmap_mermaid": result.get("mindmap_mermaid", ""),
        "ocr_result": result.get("ocr_result", ""),
        "knowledge_points": result.get("ocr_knowledge_points", []),
    }


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
    
    # 保存文件
    file_path = os.path.join(data_folder, file.filename)
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)
    
    # 增量同步
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
    """获取会话历史"""
    messages = get_messages(session_id)
    return {"session_id": session_id, "messages": messages}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 3: Commit**

```bash
git add agent/sessions.py agent.py
git commit -m "feat: add session management and main FastAPI entry with /chat, /ingest, /knowledge-structure endpoints"
```

---

### Task 8: 前端页面

**Files:**
- Create: `templates/index.html`

**Interfaces:**
- Consumes: FastAPI endpoints `/chat`, `/knowledge-structure`
- Produces: 对话界面 + OCR 上传 + 思维导图渲染

- [ ] **Step 1: 创建前端页面 `templates/index.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>小智老师 - 超级教育智能体</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f7fa;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header h1 { font-size: 20px; font-weight: 600; }
        .header .badge {
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
        }
        .main {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .message {
            margin-bottom: 16px;
            display: flex;
            gap: 12px;
        }
        .message.user { flex-direction: row-reverse; }
        .message .avatar {
            width: 36px; height: 36px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 18px; flex-shrink: 0;
        }
        .message.user .avatar { background: #667eea; }
        .message.assistant .avatar { background: #10b981; }
        .message .bubble {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .message.user .bubble { background: #667eea; color: white; }
        .message.assistant .bubble { background: white; border: 1px solid #e5e7eb; }
        .message .bubble img { max-width: 100%; border-radius: 8px; }
        .input-area {
            padding: 16px 20px;
            background: white;
            border-top: 1px solid #e5e7eb;
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }
        .input-area textarea {
            flex: 1;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 12px;
            resize: none;
            font-size: 14px;
            min-height: 44px;
            max-height: 120px;
            font-family: inherit;
        }
        .input-area textarea:focus { outline: none; border-color: #667eea; }
        .input-area button {
            width: 44px; height: 44px;
            border: none; border-radius: 12px;
            cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            font-size: 20px;
            transition: background 0.2s;
        }
        .btn-send { background: #667eea; color: white; }
        .btn-send:hover { background: #5a6fd6; }
        .btn-upload { background: #f3f4f6; color: #6b7280; }
        .btn-upload:hover { background: #e5e7eb; }
        .btn-upload input { display: none; }
        .mindmap-panel {
            width: 400px;
            background: white;
            border-left: 1px solid #e5e7eb;
            display: none;
            flex-direction: column;
            overflow: hidden;
        }
        .mindmap-panel.active { display: flex; }
        .mindmap-panel .panel-header {
            padding: 16px;
            border-bottom: 1px solid #e5e7eb;
            font-weight: 600;
        }
        .mindmap-panel .panel-body {
            flex: 1;
            overflow: auto;
            padding: 16px;
        }
        .mindmap-panel .panel-body .mermaid {
            display: flex;
            justify-content: center;
        }
        .loading { color: #9ca3af; font-style: italic; }
        .knowledge-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }
        .knowledge-tag {
            background: #ecfdf5;
            color: #059669;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 12px;
        }
        .ocr-preview {
            max-width: 200px;
            max-height: 150px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        @media (max-width: 900px) {
            .mindmap-panel { display: none !important; }
        }
    </style>
</head>
<body>
    <div class="header">
        <span style="font-size:24px">🎓</span>
        <h1>小智老师</h1>
        <span class="badge">初中教育智能体</span>
    </div>
    
    <div class="main">
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="avatar">🤖</div>
                    <div class="bubble">你好！我是小智老师，你的初中学习助手。</div>
                </div>
            </div>
            
            <div class="input-area">
                <label class="btn-upload" title="上传试卷/作业图片">
                    📷
                    <input type="file" id="imageInput" accept="image/*" multiple>
                </label>
                <textarea id="userInput" placeholder="输入你的问题，或上传试卷让老师帮你分析..." rows="1"></textarea>
                <button class="btn-send" id="sendBtn" title="发送">➤</button>
            </div>
        </div>
        
        <div class="mindmap-panel" id="mindmapPanel">
            <div class="panel-header">📊 知识点思维导图</div>
            <div class="panel-body" id="mindmapBody"></div>
        </div>
    </div>
    
    <script>
        // 初始化 Mermaid
        mermaid.initialize({ startOnLoad: true, theme: 'default' });
        
        let sessionId = localStorage.getItem('sessionId') || '';
        let pendingImages = [];
        
        if (!sessionId) {
            sessionId = crypto.randomUUID();
            localStorage.setItem('sessionId', sessionId);
        }
        
        const messagesEl = document.getElementById('messages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const imageInput = document.getElementById('imageInput');
        const mindmapPanel = document.getElementById('mindmapPanel');
        const mindmapBody = document.getElementById('mindmapBody');
        
        // Auto-resize textarea
        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
        });
        
        // Image upload
        imageInput.addEventListener('change', (e) => {
            pendingImages = Array.from(e.target.files);
            if (pendingImages.length > 0) {
                showImagePreviews();
            }
        });
        
        function showImagePreviews() {
            // Remove old previews
            document.querySelectorAll('.ocr-preview').forEach(el => el.remove());
            pendingImages.forEach((file, i) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'ocr-preview';
                    const inputArea = document.querySelector('.input-area');
                    inputArea.insertBefore(img, userInput);
                };
                reader.readAsDataURL(file);
            });
        }
        
        function addMessage(role, content, metadata = {}) {
            const msgEl = document.createElement('div');
            msgEl.className = `message ${role}`;
            
            let bubbleContent = content;
            
            // 添加知识点标签
            if (metadata.knowledge_points && metadata.knowledge_points.length > 0) {
                const tagsEl = document.createElement('div');
                tagsEl.className = 'knowledge-tags';
                metadata.knowledge_points.forEach(point => {
                    const tag = document.createElement('span');
                    tag.className = 'knowledge-tag';
                    tag.textContent = point;
                    tagsEl.appendChild(tag);
                });
                bubbleContent += '\n\n';
                bubbleContent += '<div class="knowledge-tags">';
                metadata.knowledge_points.forEach(point => {
                    bubbleContent += `<span class="knowledge-tag">${point}</span>`;
                });
                bubbleContent += '</div>';
            }
            
            // 思维导图
            if (metadata.mindmap && metadata.mindmap.trim()) {
                bubbleContent += `\n\n\`\`\`mermaid\n${metadata.mindmap}\n\`\`\``;
                setTimeout(() => {
                    renderMindmap(metadata.mindmap);
                }, 100);
            }
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = role === 'user' ? '👤' : '🤖';
            
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            bubble.innerHTML = bubbleContent;
            
            msgEl.appendChild(avatar);
            msgEl.appendChild(bubble);
            messagesEl.appendChild(msgEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }
        
        function renderMindmap(mermaidCode) {
            if (!mermaidCode || mermaidCode.trim() === '') return;
            mindmapPanel.classList.add('active');
            mindmapBody.innerHTML = `<div class="mermaid">${mermaidCode}</div>`;
            try {
                mermaid.run({ nodes: mindmapBody.querySelectorAll('.mermaid') });
            } catch (e) {
                mindmapBody.innerHTML = `<pre>${mermaidCode}</pre>`;
            }
        }
        
        async function sendMessage() {
            const text = userInput.value.trim();
            const images = pendingImages;
            
            if (!text && images.length === 0) return;
            
            // 显示用户消息
            let userBubble = text || '请分析这张图片';
            if (images.length > 0) {
                userBubble += `\n[📷 ${images.length} 张图片]`;
            }
            addMessage('user', userBubble);
            
            // 显示加载状态
            const loadingEl = document.createElement('div');
            loadingEl.className = 'message assistant';
            loadingEl.id = 'loading-msg';
            loadingEl.innerHTML = `
                <div class="avatar">🤖</div>
                <div class="bubble loading">小智老师正在思考...</div>
            `;
            messagesEl.appendChild(loadingEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;
            
            // 构建 FormData
            const formData = new FormData();
            formData.append('session_id', sessionId);
            formData.append('user_input', text || '');
            images.forEach(img => formData.append('image', img));
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // 移除加载状态
                const loadingMsg = document.getElementById('loading-msg');
                if (loadingMsg) loadingMsg.remove();
                
                // 显示回答
                const metadata = {
                    knowledge_points: data.knowledge_points || [],
                    mindmap: data.mindmap_mermaid || ''
                };
                addMessage('assistant', data.answer, metadata);
                
            } catch (error) {
                const loadingMsg = document.getElementById('loading-msg');
                if (loadingMsg) loadingMsg.remove();
                addMessage('assistant', `抱歉，处理请求时出现了错误：${error.message}`);
            }
            
            // 清空输入
            userInput.value = '';
            userInput.style.height = 'auto';
            pendingImages = [];
            imageInput.value = '';
            document.querySelectorAll('.ocr-preview').forEach(el => el.remove());
        }
        
        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add templates/
git commit -m "feat: add frontend chat interface with OCR upload and mermaid mindmap rendering"
```

---

### Task 9: 端到端测试

**Files:**
- Create: `tests/test_agent.py`

**Interfaces:**
- Consumes: FastAPI app, graph
- Produces: test coverage for all nodes

- [ ] **Step 1: 创建测试文件 `tests/test_agent.py`**

```python
"""
端到端测试 - 验证 Agent 各节点功能
"""
import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestQueryNode:
    def test_text_only_input(self):
        """测试纯文本输入"""
        from agent.nodes import query_node
        from agent.state import AgentState
        
        state = {
            "messages": [{"role": "human", "content": "什么是勾股定理？"}],
            "session_id": "test-session"
        }
        
        result = query_node(state)
        
        assert result["user_input"] == "什么是勾股定理？"
        assert result["image_data"] is None
        assert result["session_id"] == "test-session"
    
    def test_image_input(self):
        """测试带图片的输入"""
        from agent.nodes import query_node
        
        state = {
            "messages": [{
                "role": "human",
                "content": "请分析这张试卷",
                "additional_kwargs": {
                    "image_data": "data:image/jpeg;base64,abc123",
                    "session_id": "img-session"
                }
            }],
            "session_id": ""
        }
        
        result = query_node(state)
        
        assert result["user_input"] == "请分析这张试卷"
        assert result["image_data"] == "data:image/jpeg;base64,abc123"
        assert result["session_id"] == "img-session"


class TestRouteNode:
    def test_no_image_routes_to_chat(self):
        """无图片路由到 chat"""
        from agent.nodes import route_node
        
        state = {"image_data": None}
        assert route_node(state) == "chat"
    
    def test_with_image_routes_to_ocr(self):
        """有图片路由到 ocr"""
        from agent.nodes import route_node
        
        state = {"image_data": "data:image/jpeg;base64,abc"}
        assert route_node(state) == "ocr"


class TestRAGPipeline:
    def test_reciprocal_rank_fusion(self):
        """测试 RRF 融合算法"""
        from rag.pipeline import reciprocal_rank_fusion
        from langchain_core.documents import Document
        
        results = [
            [
                Document(page_content="doc1", metadata={"source": "a"}),
                Document(page_content="doc2", metadata={"source": "a"}),
            ],
            [
                Document(page_content="doc2", metadata={"source": "a"}),
                Document(page_content="doc3", metadata={"source": "b"}),
            ]
        ]
        
        fused = reciprocal_rank_fusion(results, k=30)
        
        # doc2 在两路都出现，应该排名最高
        assert len(fused) == 3
        assert fused[0].page_content == "doc2"


class TestKnowledgeStructure:
    def test_get_subjects(self):
        from rag.knowledge_structure import get_subjects
        subjects = get_subjects()
        assert "数学" in subjects
        assert "物理" in subjects
    
    def test_get_grades(self):
        from rag.knowledge_structure import get_grades
        grades = get_grades("数学")
        assert "七年级上" in grades
        assert "九年级" in grades
    
    def test_get_topics(self):
        from rag.knowledge_structure import get_topics
        topics = get_topics("数学", "九年级")
        assert "二次函数" in topics
        assert "圆" in topics


class TestVectorStore:
    def test_add_and_search(self):
        """测试向量库添加和搜索"""
        from rag.vectorstore import VectorStoreManager
        from langchain_core.documents import Document
        
        vs = VectorStoreManager(persist_directory="./test_chroma_db")
        
        docs = [
            Document(page_content="勾股定理：直角三角形中，两直角边的平方和等于斜边的平方。"),
            Document(page_content="一元一次方程：只含有一个未知数，且未知数的最高次数为1的方程。"),
        ]
        
        count = vs.add_documents(docs)
        assert count == 2
        
        results = vs.search("勾股定理", k=1)
        assert len(results) == 1
        assert "勾股定理" in results[0].page_content


class TestOCRClient:
    @pytest.mark.skip(reason="需要 Agnes API 可用")
    def test_recognize_image(self):
        """测试 OCR 识别（需要真实 API）"""
        from ocr.agnes_client import AgnesOCRClient
        
        client = AgnesOCRClient()
        # 这里需要实际的测试图片，暂时跳过
        # result = client.recognize("test_image.jpg")
        # assert isinstance(result, str)
        # assert len(result) > 0


class TestFullGraph:
    def test_build_graph(self):
        """测试图构建"""
        from agent.graph import build_graph
        
        graph = build_graph()
        assert graph is not None
        
        # 检查节点
        node_names = list(graph.nodes.keys())
        assert "query" in node_names
        assert "route" in node_names
        assert "ocr" in node_names
        assert "rag" in node_names
        assert "chat" in node_names
        assert "thought" in node_names
    
    @pytest.mark.skip(reason="需要完整 RAG 基础设施")
    def test_end_to_end_text_chat(self):
        """端到端文本对话测试"""
        from agent.graph import build_graph
        from langchain_core.messages import HumanMessage
        
        graph = build_graph()
        
        result = graph.invoke({
            "messages": [HumanMessage(content="什么是光合作用？")],
            "session_id": "test-session"
        })
        
        assert "answer" in result
        assert len(result["answer"]) > 0
    
    @pytest.mark.skip(reason="需要 Agnes API 和图片")
    def test_end_to_end_ocr_chat(self):
        """端到端 OCR + 答疑测试"""
        from agent.graph import build_graph
        from langchain_core.messages import HumanMessage
        
        graph = build_graph()
        
        result = graph.invoke({
            "messages": [HumanMessage(
                content="请分析这张试卷",
                additional_kwargs={"image_data": "test_image"}
            )],
            "session_id": "test-session"
        })
        
        assert "answer" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: 运行测试**

```bash
python -m pytest tests/test_agent.py -v --tb=short
```

预期结果：
- `TestQueryNode` - 2 passed
- `TestRouteNode` - 2 passed
- `TestRAGPipeline` - 1 passed
- `TestKnowledgeStructure` - 3 passed
- `TestVectorStore` - 1 passed（可能因嵌入模型下载慢而 timeout）
- `TestOCRClient` - 1 skipped
- `TestFullGraph` - 1 passed, 2 skipped

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test: add end-to-end tests for agent nodes, RAG pipeline, knowledge structure"
```

---

### Task 10: 启动验证

**Files:**
- None (configuration only)

**Interfaces:**
- Consumes: all previous tasks
- Produces: Running application on http://localhost:8000

- [ ] **Step 1: 启动应用**

```bash
python agent.py
```

预期输出：
```
BM25 缓存加载成功
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

- [ ] **Step 2: 验证 API 端点**

```bash
# 测试知识体系接口
curl http://localhost:8000/knowledge-structure

# 测试对话接口（纯文本）
curl -X POST http://localhost:8000/chat \
  -F "session_id=test123" \
  -F "user_input=什么是勾股定理"
```

- [ ] **Step 3: 打开浏览器访问**

访问 http://localhost:8000 确认前端页面正常加载，可以：
1. 发送文本消息进行问答
2. 上传图片进行 OCR 识别
3. 查看知识点思维导图

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "feat: complete MVP of super education agent with OCR, RAG, and mindmap"
```

---

## Verification Checklist

- [ ] `pip install -r requirements.txt` 无报错
- [ ] `python agent.py` 启动成功
- [ ] `GET /knowledge-structure` 返回学科列表
- [ ] `POST /chat` 文本问答正常返回
- [ ] 前端页面 http://localhost:8000 可访问
- [ ] 单元测试全部通过（或预期 skip）
