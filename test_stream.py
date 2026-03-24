import requests
import json
import time
from jose import jwt
from datetime import datetime, timedelta

def test_stream():
    url = "http://127.0.0.1:8000/api/v1/combined-chat/stream-answer"
    payload = {
        "question": "What is Neuroprotective agents?",
        "folder_id": "a92d748e-d903-4b29-ac91-fb0e0e59ba72",
        "history": []
    }
    
    # Generate token
    SECRET_KEY = "jwt-secret-key-change-in-production"  # Match standard local secret or pass from env
    to_encode = {"sub": "dfb89684-072e-4d31-93ee-712122a566f8", "email": "test@test.com", "role": "user"}
    to_encode.update({"exp": datetime.utcnow() + timedelta(minutes=60)})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    print("Sending POST request to stream endpoint...")
    start = time.time()
    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    print(f"[{time.time() - start:.2f}s] {line.decode('utf-8')}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_stream()
