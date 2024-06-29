import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()


async def process_request(content: str, user: dict):
    yield "LeAgent is processing your request...\n"
    await asyncio.sleep(0.5)
    yield f"LeAgent is analyzing the message from user {user['name']}: {content}\n"
    await asyncio.sleep(0.5)
    yield "LeAgent is generating a response...\n"
    await asyncio.sleep(0.5)
    yield "TASK_DONE"


@app.post("/process")
async def process(request: Request):
    data = await request.json()
    content = data["content"]
    user = data["user"]

    async def event_generator():
        async for message in process_request(content, user):
            yield f"{message}\n"

    return StreamingResponse(event_generator(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8101)
