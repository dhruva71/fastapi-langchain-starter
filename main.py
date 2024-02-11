import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core._api.deprecation import suppress_langchain_deprecation_warning
from pydantic import BaseModel

from contextlib import asynccontextmanager

import logging

from chabot.conversation import initialize_conversation_chain

with suppress_langchain_deprecation_warning():
    pass

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# file logger
handler = logging.FileHandler('chatbot.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

conversation = initialize_conversation_chain()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server started")
    yield
    logger.info("Server stopped")
    # close the logger
    for h in logger.handlers:
        h.close()
        logger.removeHandler(h)


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# CORS
origins = [
    'http://localhost:3000',
    'http://localhost:8000',
    'http://localhost:5173',
    '*',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class UserRequest(BaseModel):
    user_input: str


# Endpoint to interact with the chatbot
@app.post("/chat")
async def chat(request: UserRequest):
    logger.info(f"User input: {request.user_input}")
    user_input = request.user_input.lower()
    output = ""
    if 'bye' in user_input:
        output = conversation.invoke({"question": request.user_input})

        # Reset the conversation
        conversation.memory.clear()

        logger.info(f"Bot response: {output['text']}")
        logger.warning(f"Conversation cleared")

        return {"response": output["text"]}
    try:
        # Pass user input to the conversation chain
        output = conversation.invoke({"question": request.user_input})
        if 'bye' in output['text']:
            # Reset the conversation
            conversation.memory.clear()
            logger.warning(f"Conversation cleared")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    else:
        logger.info(f"Bot response: {output['text']}")
        # flush the logger
        for h in logger.handlers:
            h.flush()
        return {"response": output["text"]}


@app.options("/chat")
async def chat_options():
    # Allow POST and OPTIONS
    return {"Allow": "POST, OPTIONS"}


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
