"""Test query_node"""
import sys
sys.path.insert(0, r'D:\knowledge\project\ai-super-tutor')

from agent.nodes import query_node

result = query_node({'user_input': '勾股定理是什么', 'session_id': 'test', 'messages': []})
print('user_input:', repr(result['user_input']))
