import asyncio
from typing import AsyncIterable
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from starlette.staticfiles import StaticFiles

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY', 'default_value_if_not_found')


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



#app.mount("/", StaticFiles(directory=".", html=True), name="static")
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

@app.get("/hi")
def read_root():
    return {"Hello": "World"}


class Message(BaseModel):
    content: str


async def send_message(content: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
        openai_api_key=API_KEY,
        model_name="gpt-3.5-turbo",
    )

    system_message = SystemMessage(content="""
            Describe the following document with one of the following keywords:
            Mateusz, Jakub, Adam. Return the keyword and nothing else.
        """)

    human_message = HumanMessage(content=content)

    task = asyncio.create_task(
        model.agenerate(messages=[[HumanMessage(content=content)]])
        #model.agenerate(messages=[[system_message, human_message]]))
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task


@app.post("/stream_chat/")
async def stream_chat(message: Message):
    generator = send_message(message.content)
    return StreamingResponse(generator, media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app,
                port=8000,
                reload=True,
                log_level='debug'
                )
