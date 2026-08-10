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
