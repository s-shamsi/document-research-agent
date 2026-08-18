# Placeholder for agent logic.
# This will contain:
# - search_documents()
# - web_search()
# - resolve_tools() (the agent loop)
# - stream_final_answer()

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"