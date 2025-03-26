from google import genai
from dotenv import load_dotenv
from os import getenv

load_dotenv()

GEMINI_API_KEY = getenv("GEMINI_API_KEY")
from .constants import GEMINI_API_KEY
