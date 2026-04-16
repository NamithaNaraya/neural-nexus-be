"""
Quick test: hit the stream-answer endpoint with web_search=true
and print each chunk type as it arrives.
"""
import requests, json, sys

BASE = "http://localhost:8000"

# 1. Login to get token
login_resp = requests.post(f"{BASE}/api/v1/auth/login", json={
    "username": "admin",
    "password": "admin"
})

if login_resp.status_code != 200:
    print(f"Login failed {login_resp.status_code}: {login_resp.text[:200]}")
    # Try without auth
    token = None
else:
    token = login_resp.json().get("access_token") or login_resp.json().get("token")

headers = {"Content-Type": "application/json"}
if token:
    headers["Authorization"] = f"Bearer {token}"

# 2. Stream answer with web_search=true
print("\\n=== Sending stream-answer with web_search=true ===")
resp = requests.post(
    f"{BASE}/api/v1/combined-chat/stream-answer",
    json={
        "question": "What is neural nexus?",
        "web_search": True,
        "history": []
    },
    headers=headers,
    stream=True
)

print(f"Status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Error: {resp.text[:500]}")
    sys.exit(1)

chunk_types = []
for line in resp.iter_lines():
    if not line:
        continue
    decoded = line.decode('utf-8').strip()
    if not decoded:
        continue
    try:
        chunk = json.loads(decoded)
        ctype = chunk.get('type', 'unknown')
        chunk_types.append(ctype)
        if ctype == 'web_search_result':
            print(f"  ✅ WEB_SEARCH_RESULT: answer={chunk.get('data',{}).get('answer','')[:100]}...")
            print(f"     sources count: {len(chunk.get('data',{}).get('sources',[]))}")
        elif ctype == 'content':
            pass  # Too verbose
        elif ctype == 'web_search_suggestion':
            print(f"  📋 WEB_SEARCH_SUGGESTION: {chunk.get('data')}")
        elif ctype == 'step':
            print(f"  📌 STEP {chunk.get('id')}: {chunk.get('status')}")
        elif ctype == 'intent':
            print(f"  🎯 INTENT: {json.dumps(chunk.get('data',{}))[:100]}")
        else:
            print(f"  [{ctype}]: {json.dumps(chunk)[:100]}")
    except json.JSONDecodeError:
        pass

print(f"\\n=== Summary: {len(chunk_types)} chunks received ===")
print(f"Types: {dict((t, chunk_types.count(t)) for t in set(chunk_types))}")

if 'web_search_result' in chunk_types:
    print("\\n✅ Web search result WAS streamed")
else:
    print("\\n❌ Web search result was NOT streamed")
