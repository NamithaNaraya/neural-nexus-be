import requests
import json
import time
import sys

BASE = "http://localhost:8000"

def get_token():
    try:
        resp = requests.post(f"{BASE}/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin"
        })
        if resp.status_code == 200:
            return resp.json().get("access_token")
    except:
        pass
    return None

def test_latency(question):
    print(f"\nTesting Latency for: '{question}'")
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    start_time = time.time()
    ttft = None
    
    try:
        resp = requests.post(
            f"{BASE}/api/v1/combined-chat/stream-answer",
            json={
                "question": question,
                "web_search": False,
                "history": []
            },
            headers=headers,
            stream=True,
            timeout=30
        )
        
        if resp.status_code != 200:
            print(f"Error: {resp.status_code}")
            return

        for line in resp.iter_lines():
            if line:
                if ttft is None:
                    ttft = time.time() - start_time
                    print(f"⏱️ Time to First Token (TTFT): {ttft:.2f}s")
                
                decoded = line.decode('utf-8')
                try:
                    chunk = json.loads(decoded)
                    if chunk['type'] == 'content':
                        # Stop after we get some content
                        total_time = time.time() - start_time
                        print(f"✅ Content started streaming at: {total_time:.2f}s")
                        break
                except:
                    pass
                    
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_latency("Hi")
    test_latency("What is Neural Nexus?")
