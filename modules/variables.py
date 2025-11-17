import os
from google.genai import Client
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")

client = Client(api_key=GOOGLE_API_KEY)
os.environ["LANGFUSE_ENABLED"] = "True"
os.environ["LANGFUSE_ENABLED"] = "False"
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "False").lower() in ("true", "1", "t")
langfuse = None

if LANGFUSE_ENABLED:
    langfuse = Langfuse(
      secret_key=LANGFUSE_SECRET_KEY,
      public_key=LANGFUSE_PUBLIC_KEY,
      host=LANGFUSE_BASE_URL
    )