from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from agent import agent_with_db
from schemas import request
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
allowed_origins = os.getenv("ALLOWED_ORIGINS").split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
global agent
agent = agent_with_db()

@app.post("/retrieve", status_code=200)
async def retrieve(request:request):
    prev_conv = request.previous_state
    print(prev_conv)
    if prev_conv is None:
        prev_conv = "No previous conversation available, first time"
    query = request.query
    prev_conv = str(prev_conv)
    response = agent({"query": query, "previous_conversation": prev_conv})

    return {"response": response["result"]}  