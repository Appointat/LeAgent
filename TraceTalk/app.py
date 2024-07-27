import asyncio
import codecs
import os
import re
import sys
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from main import main as agent

core_directory = os.path.dirname(os.path.abspath(__file__))
if core_directory not in sys.path:
    sys.path.append(core_directory)


app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


async def process_messages(messages: List[str]):
    result = agent(messages=messages)
    chunks = re.split(r"(\n+)", result)
    for chunk in chunks:
        if chunk.strip():
            # Split the chunk into sentences
            if len(chunk) > 100:
                sentences = re.split(r"([.!?。！？]+)", chunk)
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i] + (
                        sentences[i + 1] if i + 1 < len(sentences) else ""
                    )
                    yield f"{sentence}"
                    await asyncio.sleep(0.05)  # pause between sentences
            else:
                yield f"{chunk}"

            await asyncio.sleep(0.1)  # pause between chunks
        else:
            yield chunk


@app.post("/process")
async def handler(request: Request):
    print("Received request...")
    try:
        data = await request.json()
        messages = data.get("messages", [])
        # messages_str_list = [message["content"] for message in messages
        messages_str_list = [message["content"] for message in messages]
        print(f"Received messages:\n{messages_str_list}")

        return StreamingResponse(
            process_messages(messages_str_list), media_type="text/event-stream"
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("Starting the server...")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8101)
