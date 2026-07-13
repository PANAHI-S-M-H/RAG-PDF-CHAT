import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MY_TEST_API_KEY")

print("Testing Secrets Manegement...")
print("-" * 30)
if api_key:
    print(f"Sucess! Loaded Key: {api_key}")
else:
    print("Failed To Load The Key. Check Your .env File.")

