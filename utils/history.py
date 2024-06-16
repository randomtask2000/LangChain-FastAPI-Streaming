import json
from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    role: str
    content: str


class MessagesCollection(BaseModel):
    store: List[Message] = []
    index: int = 0
    mode: str = "instruct"
    instruction_template: str = "Alpaca"
    stream: bool = True

    def add_role_content(self, role: str, content: str):
        self.store.append(Message(role=role, content=content))

    def add_message(self, message: Message):
        if isinstance(message, Message):
            self.store.append(message)
        else:
            print("Error: Invalid message type. Please add a Message object.")

    def add(self, message: Message):
        self.add_message(message)


    def to_dict(self):
        return {"history": [message.dict() for message in self.store]}

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    def get_history(self):
        return self.to_dict()['history']

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.store):
            result = self.store[self.index]
            self.index += 1
            return result
        raise StopIteration
