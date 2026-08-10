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
        embedding_model: str = r"D:\knowledge\project\pythonProject\model\models-BAAI-bge-small-zh-v1.5"
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
