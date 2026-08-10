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
