"""Test all endpoints"""
import sys
import json
import urllib.request

base_url = "http://localhost:8001"

def test_post(path, data):
    url = base_url + path
    req = urllib.request.Request(url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            print(f"POST {path}: OK")
            if 'answer' in result:
                print(f"  answer: {result['answer'][:80]}...")
            if 'mindmap_mermaid' in result and result['mindmap_mermaid']:
                print(f"  mindmap: {result['mindmap_mermaid'][:80]}...")
            return result
    except Exception as e:
        print(f"POST {path}: ERROR - {e}")
        return None

def test_get(path):
    url = base_url + path
    try:
        with urllib.request.urlopen(url) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            print(f"GET {path}: OK")
            return result
    except Exception as e:
        print(f"GET {path}: ERROR - {e}")
        return None

print("=" * 50)
print("Testing ai-super-tutor endpoints")
print("=" * 50)

# Test 1: Knowledge structure - all subjects
print("\n1. GET /knowledge-structure")
result = test_get("/knowledge-structure")
if result:
    print(f"   subjects: {result.get('subjects')}")

# Test 2: Knowledge structure - specific subject/grade
print("\n2. GET /knowledge-structure?subject=数学&grade=八年级下")
result = test_get("/knowledge-structure?subject=数学&grade=八年级下")
if result:
    print(f"   topics: {result.get('topics')}")

# Test 3: Chat - math question
print("\n3. POST /chat - 勾股定理是什么")
result = test_post("/chat", {"session_id": "test001", "user_input": "勾股定理是什么"})

# Test 4: Chat - science question
print("\n4. POST /chat - 光合作用的过程")
result = test_post("/chat", {"session_id": "test001", "user_input": "光合作用的过程"})

# Test 5: Session
print("\n5. GET /session/test001")
result = test_get("/session/test001")
if result:
    print(f"   messages: {len(result.get('messages', []))} messages")

# Test 6: Root page
print("\n6. GET /")
url = base_url + "/"
try:
    with urllib.request.urlopen(url) as resp:
        content = resp.read().decode('utf-8')
        print(f"   HTTP {resp.status}, content length: {len(content)}")
        print(f"   title: {'小智老师' in content}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 50)
print("Tests complete!")
print("=" * 50)
