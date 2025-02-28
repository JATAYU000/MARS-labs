import requests

# URL of your Flask chatbot API
URL = "http://127.0.0.1:5000/chatbot"  # Update if your server runs on a different host/port

# Hindi sentence to translate
payload = {"message": "मुझे संदेह है"}  # "I have a doubt"

# Sending a POST request
response = requests.post(URL, json=payload)

# Print the response
if response.status_code == 200:
    print("Response from chatbot:", response.json())
else:
    print("Error:", response.status_code, response.text)
