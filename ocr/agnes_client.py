"""
PaddleOCR 视觉模型 OCR 客户端
使用 PaddleOCR API（异步 job 模式）识别图片中的文字
"""
import os
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv()

JOB_URL = os.getenv("PADDLEOCR_URL", "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs")
TOKEN = os.getenv("PADDLEOCR_TOKEN", "")
MODEL = os.getenv("PADDLEOCR_MODEL", "PaddleOCR-VL-1.6")

OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": False,
}


class PaddleOCRClient:
    """PaddleOCR 异步 API 客户端"""

    def __init__(self):
        self.headers = {
            "Authorization": f"bearer {TOKEN}",
        }

    def recognize(self, image_input) -> str:
        """
        OCR 识别图片中的文字

        Args:
            image_input: bytes、文件路径或文件对象

        Returns:
            识别出的 Markdown 格式文本内容
        """
        if isinstance(image_input, str):
            # 文件路径
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图片文件不存在: {image_input}")
            return self._recognize_local_file(image_input)
        elif hasattr(image_input, 'read'):
            # 文件对象 → 保存临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp.write(image_input.read())
                tmp_path = tmp.name
            try:
                return self._recognize_local_file(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            # bytes → 保存临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp.write(image_input)
                tmp_path = tmp.name
            try:
                return self._recognize_local_file(tmp_path)
            finally:
                os.unlink(tmp_path)

    def _recognize_local_file(self, file_path: str) -> str:
        """提交本地文件到 PaddleOCR API 并等待结果"""
        # 提交 job
        data = {
            "model": MODEL,
            "opt"
            "ionalPayload": json.dumps(OPTIONAL_PAYLOAD),
        }
        with open(file_path, "rb") as f:
            files = {"file": f}
            resp = requests.post(JOB_URL, headers=self.headers, data=data, files=files, timeout=30)

        if resp.status_code != 200:
            raise Exception(f"PaddleOCR 提交失败: {resp.status_code} {resp.text}")

        job_id = resp.json()["data"]["jobId"]

        # 轮询结果
        max_wait = 60  # 最多等待 60 秒
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(2)
            elapsed += 2
            result_resp = requests.get(f"{JOB_URL}/{job_id}", headers=self.headers, timeout=10)
            if result_resp.status_code != 200:
                continue
            state = result_resp.json()["data"]["state"]
            if state == "done":
                jsonl_url = result_resp.json()["data"]["resultUrl"]["jsonUrl"]
                return self._fetch_results(jsonl_url)
            elif state == "failed":
                error_msg = result_resp.json()["data"].get("errorMsg", "未知错误")
                raise Exception(f"PaddleOCR 识别失败: {error_msg}")

        raise Exception("PaddleOCR 识别超时")

    def _fetch_results(self, jsonl_url: str) -> str:
        """从 jsonl URL 提取识别文本"""
        resp = requests.get(jsonl_url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split('\n')

        all_text = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)["result"]
                for res in result.get("layoutParsingResults", []):
                    md_text = res.get("markdown", {}).get("text", "")
                    if md_text:
                        all_text.append(md_text)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        return "\n".join(all_text).strip()


# 全局单例
_ocr_client = None


def get_ocr_client() -> PaddleOCRClient:
    """获取 OCR 客户端单例"""
    global _ocr_client
    if _ocr_client is None:
        _ocr_client = PaddleOCRClient()
    return _ocr_client
