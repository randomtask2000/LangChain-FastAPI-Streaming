import asyncio
from typing import AsyncIterable, List
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from typing import List, Optional
from pydantic import BaseModel
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from starlette.staticfiles import StaticFiles

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY', 'default_value_if_not_found')
MODEL = os.getenv('MODEL', 'gpt-3.5-turbo')


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=".", html=True), name="static")


@app.get("/hi")
def read_root():
    return {"Hello": "World"}


class Message(BaseModel):
    role: str
    content: str


class ChatHistory(BaseModel):
    messages: List[Message]


async def verify_authorization(authorization: Optional[str] = Header(None)):
    if authorization != "Bearer my_key_something":
        raise HTTPException(status_code=401, detail="Unauthorized")


async def process_tokens_history(tokens: List[Message]) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
        openai_api_key=API_KEY,
        model_name=MODEL,
    )
    new_token_list: List[BaseMessage] = []
    system_message = SystemMessage(content="""
                You are a helpful assistant named Buddy.
            """)

    new_token_list.append(system_message)

    for token in tokens:
        if token.role in ["human", "user"]:
            msg = HumanMessage(content=token.content)
        elif token.role in ["assistant", "ai"]:
            msg = AIMessage(content=token.content)
        else:
            msg = HumanMessage(content=token.content)  # Default to HumanMessage
        new_token_list.append(msg)

    task = asyncio.create_task(
        model.agenerate(messages=[new_token_list])
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
        # , authorization: Optional[str] = Depends(verify_authorization)
):
    generator = process_tokens_history(chat_history.messages)
    return StreamingResponse(generator, media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app,
                port=8000,
                reload=True,
                log_level='debug'
                )
