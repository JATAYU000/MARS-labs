import requests

BASE_URL = "http://127.0.0.1:5000"

def test_index():
    response = requests.get(f"{BASE_URL}/")
    print("Index Page:", response.status_code)

def test_get_users():
    response = requests.get(f"{BASE_URL}/users")
    print("Get All Users:", response.json())

def test_get_user_by_id(user_id):
    response = requests.get(f"{BASE_URL}/users", params={"Id": user_id})
    print(f"Get User {user_id}:", response.json())

def test_create_user():
    new_user = {
        "first_name": "John",
        "last_name": "Doe",
        "age": "25"
    }
    response = requests.post(f"{BASE_URL}/update", json=new_user)
    print("Create User Response:", response.json())

def test_chatbot_response(question):
    response = requests.get(f"{BASE_URL}/chatbot", params={"question": question})
    print(f"Chatbot Response for '{question}':", response.json())

if __name__ == "__main__":
    test_index()
    test_get_users()
    test_create_user()
    test_get_users()
    test_get_user_by_id(1)
    test_chatbot_response("Hello")
