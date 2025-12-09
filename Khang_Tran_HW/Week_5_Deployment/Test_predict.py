import requests

# API endpoint
# url = "http://localhost:8000/predict"
url = "https://fantastic-yodel-wr7pvg69wqrxc99vp-8000.app.github.dev/predict"

# JSON data to send
data = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0}

# Send POST request
response = requests.post(url, json=data)

# Print result
print("Status code:", response.status_code)
print("Response JSON:", response.json())
