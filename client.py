import requests

# URL for the prediction server
url = "http://localhost:8000/predict"

# Input image path for the captioning task
image_path = "images/bird.jpg"

# Language for the captioning task
language = "en"

# Create the payload for the request
payload = {"image_path": image_path, "language": language}

# Send the request to the server and get the response
response = requests.post(url, json=payload)
print(response.json())
