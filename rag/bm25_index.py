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
