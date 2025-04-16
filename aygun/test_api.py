import requests

# URL of the FastAPI server
url = "http://127.0.0.1:8000/predict/"

# Sample title for the request
title = "How to train a machine learning model with PyTorch?"

# Data to send (wrapped in a dictionary as per your TitleRequest model)
data = {
    "title": title
}

# Send the POST request to the FastAPI endpoint
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    result = response.json()
    print(f"Title: {result['title']}")
    print(f"Predicted Score: {result['predicted_score']}")
else:
    print(f"Error: {response.status_code}, {response.text}")
