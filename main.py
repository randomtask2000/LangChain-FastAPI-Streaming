import asyncio
from typing import AsyncIterable, List
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional
from pydantic import BaseModel
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from starlette.staticfiles import StaticFiles

from utils.history import MessagesCollection

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
    role: str
    content: str

class ChatHistory(BaseModel):
    messages: List[Message]


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
            You are a helpful assistant named Buddy.
        """)

    human_message = HumanMessage(content=content)

    task = asyncio.create_task(
        # model.agenerate(messages=[[HumanMessage(content=content)]])
        model.agenerate(messages=[[system_message, human_message]])
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


# async def verify_authorization(authorization: Optional[str] = Header(None)):
#     # if authorization != "Bearer my_key_something": # should be !=
#     #     raise HTTPException(status_code=401, detail="Unauthorized")


async def process_message_history(messages: List[Message]) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
        openai_api_key=API_KEY,
        model_name="gpt-3.5-turbo",
    )
    msgs: List[BaseMessage] = []
    system_message = SystemMessage(content="""
                You are a helpful assistant named Buddy.
            """)

    msgs.append(system_message)

    for message in messages:
        if message.role in ["human", "user"]:
            msg = HumanMessage(content=message.content)
        elif message.role in ["assistant", "ai"]:
            msg = AIMessage(content=message.content)
        else:
            msg = HumanMessage(content=message.content)  # Default to HumanMessage
        msgs.append(msg)

    for message in messages:
        if message.role in ["human", "user"]:
            msg = HumanMessage(content=message.content)
        else:
            msg = HumanMessage(content=message.content)
        msgs.append(msg)

    task = asyncio.create_task(
        model.agenerate(messages=[msgs])
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task


@app.post("/stream_history/")
async def stream_history(
        chat_history: ChatHistory
        # ,
        # authorization: Optional[str] = Depends(verify_authorization)
):
    generator = process_message_history(chat_history.messages)
    return StreamingResponse(generator, media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app,
                port=8000,
                reload=True,
                log_level='debug'
                )
