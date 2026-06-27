import requests
import json

# Credentials
HANDLE = "paulus-ai.bsky.social"
APP_PASSWORD = "if2j-yayc-77cv-s7q3"

def read_feed():
    session_url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    payload = {"identifier": HANDLE, "password": APP_PASSWORD}
    
    try:
        response = requests.post(session_url, json=payload)
        response.raise_for_status()
        session = response.json()
        access_token = session['accessJwt']
        
        timeline_url = "https://bsky.social/xrpc/app.bsky.feed.getTimeline"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"limit": 5}
        
        res = requests.get(timeline_url, headers=headers, params=params)
        res.raise_for_status()
        return res.json() # Return the whole object to debug
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    data = read_feed()
    print(json.dumps(data, indent=2))
