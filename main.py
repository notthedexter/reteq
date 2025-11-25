from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.chatbot.chatbot_route import router as chatbot_router

app = FastAPI(
    title="SocialWiz AI - Unified Chatbot API",
    description="Unified API with all three modes in one endpoint: Message Rewriter, Icebreaker Generator, and Curveball Handler with optional image upload support.",
    version="2.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include unified chatbot router (ONLY ENDPOINT)
app.include_router(chatbot_router, prefix="/api/chatbot", tags=["Chatbot"])

@app.get("/")
async def root():
    return {
        "message": "SocialWiz AI - Your Conversation Assistant",
        "version": "2.0.0",
        "endpoint": "/api/chatbot/chat",
        "description": "Unified endpoint with all three modes - select via 'mode' field",
        "modes": {
            "mode1": {
                "name": "Message Rewriter",
                "description": "Rewrite messages with different moods",
                "moods": ["casual", "flirty", "nonchalant", "slap_back"],
                "required_fields": ["original_message", "response"],
                "optional_fields": ["mood", "personal_context", "file", "image_base64"],
                "supports_image": True
            },
            "mode2": {
                "name": "Icebreaker Generator",
                "description": "Generate conversation openers",
                "opener_types": ["if_you", "between", "if_i_was", "would_you_rather", "what_is_something", "in_a_situation", "how_would_you_handle"],
                "required_fields": ["opener_type"],
                "optional_fields": ["context"],
                "supports_image": False
            },
            "mode3": {
                "name": "Curveball Handler",
                "description": "Handle awkward situations",
                "moods": ["casual", "apologetic", "encouraging", "sarcastic", "curious", "flirty", "empathetic"],
                "required_fields": ["situation_description"],
                "optional_fields": ["mood", "file", "image_base64"],
                "supports_image": True
            }
        },
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
