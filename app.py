import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from context import TWIN_SYSTEM_PROMPT
from tools import tools, handle_tool_calls
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL_NAME = "gpt-5.4"

openai_client = OpenAI()

system = [{"role": "system", "content": TWIN_SYSTEM_PROMPT}]

app = FastAPI()

# Allow your frontend to make requests to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://siddartha.dev",
        "https://www.siddartha.dev",
        "https://digitaltwin.siddartha.dev",
        "http://localhost:5173",  # Vite dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    return FileResponse("index.html")


@app.get("/owl.png")
async def owl():
    return FileResponse("owl.png")


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data["message"]
    history = data.get("history", [])

    messages = system + history + [{"role": "user", "content": message}]
    response = openai_client.chat.completions.create(
        model=MODEL_NAME, messages=messages, tools=tools
    )

    while response.choices[0].finish_reason == "tool_calls":
        msg = response.choices[0].message
        tool_calls = msg.tool_calls
        results = handle_tool_calls(tool_calls)
        messages.append(msg)
        messages.extend(results)
        response = openai_client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools
        )

    return {"response": response.choices[0].message.content}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
