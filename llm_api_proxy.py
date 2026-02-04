"""
Simple OpenAI-compatible API proxy for Ollama
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import requests
import uvicorn

app = FastAPI(title="Local LLM API Proxy", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama2")

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-fake"
    object: str = "chat.completion"
    created: int = 1234567890
    model: str
    choices: List[Choice]
    usage: Usage

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Convert OpenAI-style request to Ollama request"""
    # Determine model to use
    model = request.model or MODEL_NAME
    
    # Format messages for Ollama
    ollama_messages = []
    for msg in request.messages:
        ollama_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Prepare Ollama request
    ollama_request = {
        "model": model,
        "messages": ollama_messages,
        "options": {
            "temperature": request.temperature,
        },
        "stream": request.stream
    }
    
    if request.max_tokens:
        ollama_request["options"]["num_predict"] = request.max_tokens
    
    # Call Ollama API
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=ollama_request
        )
        response.raise_for_status()
        
        ollama_result = response.json()
        
        # Convert Ollama response to OpenAI format
        content = ollama_result.get("message", {}).get("content", "")
        
        chat_response = ChatCompletionResponse(
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=content),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=0,  # Ollama doesn't provide this directly
                completion_tokens=0,  # Approximate or estimate
                total_tokens=0
            )
        )
        
        return chat_response
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@app.get("/v1/models")
async def list_models():
    """List available models (mock response)"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1234567890,
                "owned_by": "user"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)