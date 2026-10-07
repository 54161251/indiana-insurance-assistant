import os
import requests

# Replace with your actual OpenRouter API key
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# API endpoint (for OpenRouter; replace with specific model endpoint if needed)
url = "https://openrouter.ai/api/v1/chatJI"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# A simple example message
data = {
    "model": "google/gemma-3n-e2b-it:free",    # You can change to other supported models
    "messages": [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the meaning of life?"},
    ],
    "max_tokens": 1000,
    "temperature": 0.7,
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    print("Assistant:", result["choices"][0]["message"]["content"])
else:
    print("Error:", response.status_code, response.text)
