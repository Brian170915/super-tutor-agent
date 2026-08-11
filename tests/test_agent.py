"""
端到端测试 - 验证 Agent 各节点功能
"""
import os
import sys
import pytest
from langchain_core.messages import HumanMessage, AIMessage

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestQueryNode:
    def test_text_only_input(self):
        """测试纯文本输入"""
        from agent.nodes import query_node

        state = {
            "messages": [HumanMessage(content="什么是勾股定理？")],
            "session_id": "test-session"
        }

        result = query_node(state)

        assert result["user_input"] == "什么是勾股定理？"
        assert result["session_id"] == "test-session"

    def test_dict_input(self):
        """测试 dict 输入"""
        from agent.nodes import query_node

        state = {
            "user_input": "勾股定理是什么",
            "session_id": "test-session-2",
            "messages": []
        }

        result = query_node(state)

        assert result["user_input"] == "勾股定理是什么"
        assert result["session_id"] == "test-session-2"
        # 验证消息追加到历史
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "勾股定理是什么"

    def test_turn_count_preserved(self):
        """测试 turn_count 不被重置"""
        from agent.nodes import query_node

        state = {
            "user_input": "测试问题",
            "session_id": "test-session-3",
            "messages": [],
            "turn_count": 5,
        }

        result = query_node(state)

        # turn_count 应该被保留，而不是重置为 0
        assert result["turn_count"] == 5
        # intent 应该被初始化为空
        assert result["intent"] == ""


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
        import shutil
        from rag.vectorstore import VectorStoreManager
        from langchain_core.documents import Document

        # 使用固定目录避免 Windows 临时目录锁文件问题
        test_dir = "./test_chroma_db_temp"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        try:
            vs = VectorStoreManager(persist_directory=test_dir)

            docs = [
                Document(page_content="勾股定理：直角三角形中，两直角边的平方和等于斜边的平方。"),
                Document(page_content="一元一次方程：只含有一个未知数，且未知数的最高次数为1的方程。"),
            ]

            count = vs.add_documents(docs)
            assert count == 2

            results = vs.search("勾股定理", k=1)
            assert len(results) == 1
            assert "勾股定理" in results[0].page_content
        finally:
            # 清理测试数据
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=True)


class TestContextManager:
    """测试上下文管理器"""

    def test_estimate_tokens(self):
        """测试 Token 估算"""
        from agent.context_manager import estimate_tokens

        # 空文本
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

        # 中文文本
        text = "你好世界"
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_manage_history_short(self):
        """测试短消息历史（不触发摘要）"""
        from agent.context_manager import ContextManager

        cm = ContextManager()
        messages = [
            HumanMessage(content="你好"),
            AIMessage(content="你好！有什么可以帮你的？"),
            HumanMessage(content="什么是勾股定理？"),
        ]

        managed, summary, token_count = cm.manage_history(messages)

        # 消息较少，不应触发摘要
        assert summary == ""
        assert len(managed) == 3
        assert token_count > 0

    def test_manage_history_long(self):
        """测试长消息历史（触发摘要）"""
        from agent.context_manager import ContextManager

        cm = ContextManager()
        # 创建超过 SUMMARIZE_THRESHOLD 的消息
        messages = []
        for i in range(15):
            messages.append(HumanMessage(content=f"问题{i}: 这是第{i}个问题"))
            messages.append(AIMessage(content=f"回答{i}: 这是第{i}个回答"))

        managed, summary, token_count = cm.manage_history(messages)

        # 应该触发摘要
        assert summary != ""
        # 保留最近 WINDOW_SIZE 条
        assert len(managed) <= 12  # 摘要 + 窗口消息

    def test_assemble_context(self):
        """测试上下文组装"""
        from agent.context_manager import ContextManager

        cm = ContextManager()
        messages = [HumanMessage(content="测试消息")]
        rag_context = "这是 RAG 上下文"

        result = cm.assemble_context(messages, rag_context=rag_context)

        assert "messages" in result
        assert "token_count" in result
        assert "truncated" in result
        assert "context_summary" in result
        assert result["token_count"] > 0


class TestCrossTurnTracker:
    """测试跨轮次追踪器"""

    def test_get_or_create(self):
        """测试获取或创建追踪器"""
        from agent.context_tracker import CrossTurnTracker

        tracker = CrossTurnTracker()
        session_tracker = tracker.get_or_create("test-session")

        assert session_tracker.session_id == "test-session"
        assert session_tracker.turn_count == 0
        assert session_tracker.subject_history == []

    def test_cleanup(self):
        """测试清理追踪器"""
        from agent.context_tracker import CrossTurnTracker

        tracker = CrossTurnTracker()
        tracker.get_or_create("test-session")
        tracker.cleanup("test-session")

        # 再次获取应该创建新的
        new_tracker = tracker.get_or_create("test-session")
        assert new_tracker.session_id == "test-session"
        assert new_tracker.turn_count == 0

    def test_to_dict(self):
        """测试序列化"""
        from agent.context_tracker import SessionTracker

        st = SessionTracker("test-session")
        st.subject_history = ["数学"]
        st.current_topics = ["勾股定理"]
        st.turn_count = 5

        d = st.to_dict()
        assert d["session_id"] == "test-session"
        assert d["subject_history"] == ["数学"]
        assert d["turn_count"] == 5

    def test_detect_gaps_rule_based(self):
        """测试基于规则的困难点检测"""
        from agent.context_tracker import CrossTurnTracker

        tracker = CrossTurnTracker()
        messages = [
            HumanMessage(content="what is the pythagorean theorem?"),
            AIMessage(content="The Pythagorean theorem is..."),
            HumanMessage(content="I still don't understand"),
            HumanMessage(content="can you explain it again?"),
        ]

        gaps = tracker.detect_gaps(messages)
        # 应该检测到困惑表达
        assert len(gaps) > 0

    def test_identify_topics_no_llm(self):
        """测试无 LLM 时的主题识别"""
        from agent.context_tracker import CrossTurnTracker

        tracker = CrossTurnTracker()
        messages = [
            HumanMessage(content="勾股定理是什么？"),
        ]

        result = tracker.identify_topics(messages, llm=None)
        assert result == {"subjects": [], "topics": []}


class TestRAGScoring:
    """测试 RAG 相关性评分"""

    def test_score_and_filter_empty(self):
        """测试空文档列表"""
        from agent.nodes import _score_and_filter_docs

        docs, scores = _score_and_filter_docs([], "测试问题")
        assert docs == []
        assert scores == []

    def test_compress_docs(self):
        """测试文档压缩"""
        from agent.nodes import _compress_docs

        docs = [
            "这是第一个文档的内容，比较长...",
            "这是第二个文档的内容，也比较长...",
            "这是第一个文档的内容，比较长..."  # 重复
        ]

        compressed = _compress_docs(docs, max_chars_per_doc=50)

        # 应该去重
        assert len(compressed) < len(docs)
        # 应该截断
        for doc in compressed:
            assert len(doc) <= 53  # 50 + "..."


class TestOCRClient:
    @pytest.mark.skip(reason="需要 PaddleOCR API 可用")
    def test_ocr_client_exists(self):
        """测试 PaddleOCR 客户端可导入"""
        from ocr.agnes_client import PaddleOCRClient

        client = PaddleOCRClient()
        assert client.headers["Authorization"].startswith("bearer ")
        assert client.model == "PaddleOCR-VL-1.6"


class TestFullGraph:
    def test_build_graph(self):
        """测试图构建"""
        from agent.graph import build_graph

        graph = build_graph()
        assert graph is not None

        # 检查节点 - 现在有 6 个节点（新增 intent）
        node_names = list(graph.nodes.keys())
        assert "query" in node_names
        assert "intent" in node_names
        assert "context_manager" in node_names
        assert "rag" in node_names
        assert "chat" in node_names
        assert "thought" in node_names
        # route 和 ocr 节点已移除
        assert "route" not in node_names
        assert "ocr" not in node_names

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


class TestRouting:
    """测试意图路由函数"""

    def test_route_by_intent_explain(self):
        """explain 意图路由到 rag"""
        from agent.nodes import route_by_intent
        assert route_by_intent({"intent": "explain"}) == "rag"

    def test_route_by_intent_unknown(self):
        """unknown 意图路由到 rag（降级）"""
        from agent.nodes import route_by_intent
        assert route_by_intent({"intent": "unknown"}) == "rag"

    def test_route_by_intent_quiz(self):
        """quiz 意图路由到 chat"""
        from agent.nodes import route_by_intent
        assert route_by_intent({"intent": "quiz"}) == "chat"

    def test_route_by_intent_summary(self):
        """summary 意图路由到 chat"""
        from agent.nodes import route_by_intent
        assert route_by_intent({"intent": "summary"}) == "chat"

    def test_route_by_intent_chat(self):
        """chat 意图路由到 chat"""
        from agent.nodes import route_by_intent
        assert route_by_intent({"intent": "chat"}) == "chat"

    def test_route_after_chat_thought(self):
        """explain 意图 chat 后路由到 thought"""
        from agent.nodes import route_after_chat
        assert route_after_chat({"intent": "explain"}) == "thought"

    def test_route_after_chat_end_quiz(self):
        """quiz 意图 chat 后直接结束"""
        from agent.nodes import route_after_chat
        from langgraph.graph import END
        assert route_after_chat({"intent": "quiz"}) == END

    def test_route_after_chat_end_summary(self):
        """summary 意图 chat 后直接结束"""
        from agent.nodes import route_after_chat
        from langgraph.graph import END
        assert route_after_chat({"intent": "summary"}) == END


class TestContextWindow:
    """测试上下文窗口管理"""

    def test_token_budget_enforcement(self):
        """测试 Token 预算控制"""
        from agent.context_manager import ContextManager, CONTEXT_CONFIG

        cm = ContextManager()
        assert cm.config["max_total_tokens"] == CONTEXT_CONFIG["max_total_tokens"]

    def test_truncation_behavior(self):
        """测试截断行为"""
        from agent.context_manager import ContextManager

        cm = ContextManager()
        # 创建超长消息
        long_messages = [HumanMessage(content="测试" * 1000) for _ in range(10)]

        result = cm.assemble_context(long_messages, rag_context="短上下文")
        assert result["truncated"] == True
        assert len(result["messages"]) < len(long_messages)


class TestQuizPrompt:
    """测试批量出题 Prompt"""

    def test_quiz_batch_prompt_exists(self):
        """测试 QUIZ_BATCH_PROMPT 已定义"""
        from agent.prompts import QUIZ_BATCH_PROMPT
        assert QUIZ_BATCH_PROMPT is not None
        assert hasattr(QUIZ_BATCH_PROMPT, 'input_variables')

    def test_quiz_batch_prompt_variables(self):
        """测试 Prompt 变量"""
        from agent.prompts import QUIZ_BATCH_PROMPT
        expected_vars = {"topics", "gaps", "subjects", "history", "count"}
        actual_vars = set(QUIZ_BATCH_PROMPT.input_variables)
        assert expected_vars.issubset(actual_vars)


class TestFeatureEndpoints:
    """测试功能面板相关端点结构"""

    def _load_root_agent(self):
        """加载根目录的 agent.py 模块（避免与 agent/ 包冲突）"""
        import importlib.util
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        agent_path = os.path.join(project_root, "agent.py")
        spec = importlib.util.spec_from_file_location("root_agent", agent_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_quiz_request_model(self):
        """测试出题请求模型"""
        mod = self._load_root_agent()
        req = mod.QuizRequest(session_id="test-123", count=5)
        assert req.session_id == "test-123"
        assert req.count == 5

    def test_chat_request_with_image(self):
        """测试带图片的聊天请求"""
        mod = self._load_root_agent()
        req = mod.ChatRequest(session_id="test", user_input="你好", image_base64="dGVzdA==")
        assert req.image_base64 == "dGVzdA=="

    def test_chat_request_without_image(self):
        """测试不带图片的聊天请求"""
        mod = self._load_root_agent()
        req = mod.ChatRequest(session_id="test", user_input="你好")
        assert req.image_base64 == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
