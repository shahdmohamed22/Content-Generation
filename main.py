from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel, Field
from app.generate import generate_text
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Generation System",
    version="1.0.0",
    description="API for generating text using a fine-tuned GPT-2 model",
    openapi_tags=[{
        "name": "Generation",
        "description": "Text generation endpoints"
    }]
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class GenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        example="Tell me a story about a prince and a scientist",
        description="Input prompt for generation"
    )
    max_new_tokens: int = Field(
        100,
        ge=20,
        le=500,
        example=100,
        description="Number of tokens to generate (20-500)"
    )
    temperature: float = Field(
        0.9,
        ge=0.1,
        le=1.5,
        example=0.9,
        description="Control the creativity of the response"
    )
    top_p: float = Field(
        0.95,
        ge=0.1,
        le=1.0,
        example=0.95,
        description="Focus on the highest probability words"
    )
    repetition_penalty: float = Field(
        1.5,
        ge=1.0,
        le=2.0,
        example=1.5,
        description="Penalty for repetition in generated text"
    )

class GenerationResponse(BaseModel):
    status: str = Field(..., example="success")
    generated_text: str = Field(..., example="The generated text will appear here...")
    parameters: dict = Field(..., example={"prompt": "...", "max_new_tokens": 100})

# Endpoints
@app.post(
    "/generate",
    tags=["Generation"],
    response_model=GenerationResponse,
    responses={
        200: {"description": "Successful text generation"},
        400: {"description": "Invalid parameters"},
        500: {"description": "Generation error"}
    },
    summary="Generate text",
    description="Generate text using a fine-tuned GPT-2 model with configurable parameters."
)
async def generate_text_endpoint(request: GenerationRequest = Body(...)):
    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
            
        logger.info(f"Generation request for prompt: {request.prompt[:50]}...")

        generated_text = generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=50,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty
        )
        
        return {
            "status": "success",
            "generated_text": generated_text,
            "parameters": request.dict()
        }

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=500, detail="GPU memory exhausted")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "healthy"}



