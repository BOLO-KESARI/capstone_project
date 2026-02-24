import os
from dotenv import load_dotenv

load_dotenv()

API_ACCESS_KEY = os.getenv("AVIATIONSTACK_API_KEY", "YOUR_ACCESS_KEY")
BASE_URL = "http://api.aviationstack.com/v1"
