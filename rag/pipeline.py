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
