"""Debug graph execution"""
import sys
sys.path.insert(0, r'D:\knowledge\project\ai-super-tutor')

from agent.graph import build_graph

graph = build_graph()
print("Graph built successfully, nodes:", list(graph.nodes.keys()))

try:
    result = graph.invoke({
        "user_input": "勾股定理是什么",
        "session_id": "test",
        "messages": [],
    })
    print("Answer:", result.get("answer", "")[:200])
    print("Mindmap:", result.get("mindmap_mermaid", "")[:100])
except Exception as e:
    import traceback
    traceback.print_exc()
