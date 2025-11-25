from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Dict, Optional
import os
import json
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai
from PIL import Image
import io
from enum import Enum

# Load environment variables
load_dotenv()

router = APIRouter()

# API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_VISION_MODEL = genai.GenerativeModel('gemini-2.0-flash-lite')
GEMINI_TEXT_MODEL = genai.GenerativeModel('gemini-2.0-flash-lite')


# ============================================================================
# MODE 1: MESSAGE REWRITER
# ============================================================================

class Mode1Mood(str, Enum):
    CASUAL = "casual"
    FLIRTY = "flirty"
    NONCHALANT = "nonchalant"
    SLAP_BACK = "slap_back"


def build_mode1_prompt(original_message: str, user_response: str, mood: str, personal_context: Optional[str] = None, image_context: Optional[str] = None) -> str:
    """Build prompt for Mode1: Message Rewriter"""
    
    mood_instructions = {
        "casual": "Casual, friendly, and relaxed.",
        "flirty": "Subtly playful and flirty (not explicit).",
        "nonchalant": "Short, detached, and unconcerned.",
        "slap_back": "Witty, biting, and sharp (not abusive)."
    }
    
    target_tone = mood_instructions.get(mood, "Match the requested tone.")
    
    ctx_text = f"\n- Personal Context: {personal_context}" if personal_context else ""
    img_text = f"\n- Visual Context from Chat Screenshot: {image_context}" if image_context else ""
    
    prompt = f"""
SYSTEM INSTRUCTIONS:
You are an expert conversation stylist. Your task is to rewrite a user's draft reply to match a specific target mood.
You must strictly follow these rules:
1. Analyze the 'Original Message' and 'User Draft'.
2. Rewrite the 'User Draft' to match the 'Target Tone'.
3. Incorporate 'Personal Context' if provided.
4. Use 'Visual Context' from the chat screenshot to better understand the conversation flow and dynamics.
5. OUTPUT FORMAT: Respond ONLY with a JSON object in this exact format: {{"reply": "your rewritten message here"}}
6. Do NOT include markdown formatting, explanations, or any other text.

INPUT DATA:
- Original Message: "{original_message}"
- User Draft: "{user_response}"
- Target Tone: {target_tone}{ctx_text}{img_text}

Respond with ONLY: {{"reply": "your rewritten message"}}
"""
    return prompt.strip()


# ============================================================================
# MODE 2: ICEBREAKER GENERATOR
# ============================================================================

class Mode2OpenerType(str, Enum):
    IF_YOU = "if_you"
    BETWEEN = "between"
    IF_I_WAS = "if_i_was"
    WOULD_YOU_RATHER = "would_you_rather"
    WHAT_IS_SOMETHING = "what_is_something"
    IN_A_SITUATION = "in_a_situation"
    HOW_WOULD_YOU_HANDLE = "how_would_you_handle"


def build_mode2_prompt(opener_type: str, context: Optional[str] = None) -> str:
    """Build prompt for Mode2: Icebreaker Generator"""
    
    opener_type_instructions = {
        "if_you": "Start with 'If you...' followed by an interesting hypothetical scenario or question.",
        "between": "Start with 'Between...' presenting two interesting options or choices.",
        "if_i_was": "Start with 'If I was...' creating an imaginative scenario or role-play situation.",
        "would_you_rather": "Start with 'Would you rather...' presenting two intriguing alternatives.",
        "what_is_something": "Start with 'What is something that...' asking about preferences, experiences, or beliefs.",
        "in_a_situation": "Start with 'In a situation where...' describing a hypothetical scenario.",
        "how_would_you_handle": "Start with 'How would you handle...' presenting a situation requiring a response or decision."
    }
    
    opener_instruction = opener_type_instructions.get(opener_type, "Generate an engaging conversation opener.")
    ctx_text = f"\nCONTEXT: {context}" if context else ""
    
    prompt = f"""
SYSTEM INSTRUCTIONS:
You are an expert conversation starter and social communication specialist. Your task is to generate engaging, creative, and personalized icebreaker messages for conversations.

You must strictly follow these rules:
1. Generate a conversation opener based on the specified 'Opener Type'.
2. Make it interesting, engaging, and natural - not too formal or awkward.
3. If 'Context' is provided, incorporate it subtly to make the opener more personalized and relevant.
4. Keep it concise (1-3 sentences maximum).
5. Make it open-ended to encourage a response.
6. OUTPUT FORMAT: Respond ONLY with a JSON object in this exact format: {{"reply": "your icebreaker message here"}}
7. Do NOT include markdown formatting, explanations, or any other text.

TASK:
- Opener Type: {opener_instruction}{ctx_text}

Respond with ONLY: {{"reply": "your icebreaker message"}}
"""
    return prompt.strip()


# ============================================================================
# MODE 3: CURVEBALL HANDLER
# ============================================================================

class Mode3Mood(str, Enum):
    CASUAL = "casual"
    APOLOGETIC = "apologetic"
    ENCOURAGING = "encouraging"
    SARCASTIC = "sarcastic"
    CURIOUS = "curious"
    FLIRTY = "flirty"
    EMPATHETIC = "empathetic"


def build_mode3_prompt(situation_description: str, mood: str, image_context: Optional[str] = None) -> str:
    """Build prompt for Mode3: Curveball Handler"""
    
    mood_instructions = {
        "casual": "Keep it light, relaxed, and easy-going. Don't overthink the situation.",
        "apologetic": "Be understanding and apologetic. Show you care about the confusion or misunderstanding.",
        "encouraging": "Be positive, supportive, and uplifting. Turn the awkward moment into something good.",
        "sarcastic": "Use witty sarcasm and humor to deflect or address the situation playfully (not mean).",
        "curious": "Show genuine interest and curiosity. Ask questions to understand better.",
        "flirty": "Turn the awkward moment into a playful, flirty opportunity while staying smooth.",
        "empathetic": "Show deep understanding and emotional connection. Validate their feelings or situation."
    }
    
    target_tone = mood_instructions.get(mood, "Handle the situation appropriately.")
    img_text = f"\n- Visual Context from Screenshot: {image_context}" if image_context else ""
    
    prompt = f"""
SYSTEM INSTRUCTIONS:
You are an expert at handling awkward, confusing, or curveball moments in conversations. Your task is to craft the perfect reply that handles the situation smoothly.

You must strictly follow these rules:
1. Analyze the curveball/awkward situation described.
2. Generate a reply that handles it according to the specified mood.
3. Make it natural, authentic, and conversation-appropriate.
4. Keep it concise (1-3 sentences typically).
5. Turn the awkward moment into a smooth continuation of the conversation.
6. OUTPUT FORMAT: Respond ONLY with a JSON object in this exact format: {{"reply": "your response here"}}
7. Do NOT include markdown formatting, explanations, or any other text.

INPUT DATA:
- Situation: "{situation_description}"
- Target Mood: {target_tone}{img_text}

Respond with ONLY: {{"reply": "your curveball response"}}
"""
    return prompt.strip()


async def extract_context_from_image(image: Image.Image, context_type: str = "general") -> Optional[str]:
    """
    Extract conversation context from chat screenshot using Gemini Vision.
    
    Args:
        image: PIL Image object
        context_type: "general" for Mode1, "curveball" for Mode3
    """
    try:
        
        # Adjust prompt based on context type
        if context_type == "curveball":
            prompt = """Analyze this chat screenshot image carefully.

STEP 1 - Extract messages by spatial position:
- Messages on the LEFT side are from the OTHER PERSON
- Messages on the RIGHT side are from ME (the user)

STEP 2 - Understand the conversation context:
- What is the conversation about?
- What is the tone/mood of the conversation?
- What is the relationship dynamic between the two people?
- What topics are being discussed?
- Is there any emotional undertone (friendly, flirty, serious, casual, conflicted, etc.)?
- What is the awkward or curveball situation visible?

Provide a clear, concise description (2-3 sentences) of the conversation context, emotional dynamics, and the curveball situation.
Do NOT include message sequences, timestamps, or technical details.
Do NOT list individual messages.
Return plain text description only."""
        else:
            prompt = """Analyze this chat screenshot image carefully.

STEP 1 - Extract messages by spatial position:
- Messages on the LEFT side are from the OTHER PERSON
- Messages on the RIGHT side are from ME (the user)

STEP 2 - Understand the conversation context:
- What is the conversation about?
- What is the tone/mood of the conversation?
- What is the relationship dynamic between the two people?
- What topics are being discussed?
- Is there any emotional undertone (friendly, flirty, serious, casual, conflicted, etc.)?

Provide a clear, concise description (2-3 sentences) of the conversation context.
Do NOT include message sequences, timestamps, or technical details.
Do NOT list individual messages.
Return plain text description only."""
        
        # Configure generation settings
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Generate response with image and prompt
        response = GEMINI_VISION_MODEL.generate_content(
            [prompt, image],
            generation_config=generation_config
        )
        
        if response.text:
            return response.text.strip()
        else:
            return None
            
    except Exception as e:
        print(f"Image context extraction failed: {str(e)}")
        return None





async def call_groq_api(prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """Call Groq API for text generation."""
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if completion.choices[0].message.content:
            return completion.choices[0].message.content.strip()
        else:
            raise HTTPException(status_code=500, detail="Groq API returned empty response")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to call Groq API: {str(e)}")


async def call_gemini_api(prompt: str, temperature: float = 0.8) -> str:
    """Call Gemini API for text generation."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        response = model.generate_content(prompt, generation_config=generation_config)
        
        if response.text:
            return response.text.strip()
        else:
            raise HTTPException(status_code=500, detail="Gemini API returned empty response")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to call Gemini API: {str(e)}")


def parse_json_response(llm_output: str) -> Dict:
    """Parse LLM response, handling markdown code blocks"""
    cleaned_output = llm_output.strip()
    
    # Remove markdown code blocks
    if cleaned_output.startswith("```json"):
        cleaned_output = cleaned_output[7:]
    elif cleaned_output.startswith("```"):
        cleaned_output = cleaned_output[3:]
    
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-3]
    
    try:
        return json.loads(cleaned_output.strip())
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response: {str(e)}", "raw_output": llm_output}


@router.post("/chat", response_model=Dict)
async def unified_chat(
    mode: str = Form(..., description="Mode selection: mode1, mode2, or mode3"),
    
    # Mode1 fields
    original_message: Optional[str] = Form(None, description="[Mode1] The original message from the other person"),
    response: Optional[str] = Form(None, description="[Mode1] Your draft reply"),
    mood: Optional[str] = Form(None, description="[Mode1/Mode3] Mood/tone (e.g., casual, flirty, nonchalant, slap_back for Mode1; casual, apologetic, encouraging, sarcastic, curious, flirty, empathetic for Mode3)"),
    personal_context: Optional[str] = Form(None, description="[Mode1] Optional personal context"),
    
    # Mode2 fields
    opener_type: Optional[str] = Form(None, description="[Mode2] Opener type (if_you, between, if_i_was, would_you_rather, what_is_something, in_a_situation, how_would_you_handle)"),
    context: Optional[str] = Form(None, description="[Mode2] Optional context for icebreaker"),
    
    # Mode3 fields
    situation_description: Optional[str] = Form(None, description="[Mode3] Description of the curveball/awkward situation"),
    
    # Image uploads (optional for Mode1 and Mode3)
    file: UploadFile= File(None, description="Optional: Chat screenshot image file (PNG, JPEG, WEBP, HEIC, HEIF) - for Mode1 and Mode3")
):
    """
    Unified chatbot endpoint supporting all three modes:
    
    - **Mode1 (Message Rewriter)**: Rewrite messages with different moods. Supports optional image upload.
    - **Mode2 (Icebreaker Generator)**: Generate conversation openers with various patterns.
    - **Mode3 (Curveball Handler)**: Handle awkward situations with different moods. Supports optional image upload.
    """
    try:
        mode = mode.lower()
        
        # Extract image context if provided (Mode1 or Mode3)
        image_context = None
        
        if mode in ["mode1", "mode3"] and file:
            context_type = "curveball" if mode == "mode3" else "general"
            
            # Validate file MIME type
            allowed_types = ["image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"]
            if not file.content_type or file.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File must be one of: PNG, JPEG, WEBP, HEIC, HEIF. Received: {file.content_type}"
                )
            
            # Read file and convert to PIL Image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Extract context using PIL Image
            image_context = await extract_context_from_image(image, context_type)
        
        # MODE 1: Message Rewriter
        if mode == "mode1":
            if not original_message or not response:
                raise HTTPException(
                    status_code=400,
                    detail="Mode1 requires 'original_message' and 'response' fields"
                )
            
            # Build prompt
            prompt = build_mode1_prompt(
                original_message=original_message,
                user_response=response,
                mood=mood or "casual",
                personal_context=personal_context,
                image_context=image_context
            )
            
            # Call Gemini API
            raw_response = await call_gemini_api(prompt, temperature=0.7)
            
            # Parse response
            result = parse_json_response(raw_response)
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=f"Response parsing failed: {result['error']}")
            
            if "reply" not in result:
                raise HTTPException(status_code=500, detail="Missing 'reply' in response")
            
            return {"reply": result["reply"]}
        
        # MODE 2: Icebreaker Generator
        elif mode == "mode2":
            if not opener_type:
                raise HTTPException(
                    status_code=400,
                    detail="Mode2 requires 'opener_type' field"
                )
            
            # Build prompt
            prompt = build_mode2_prompt(
                opener_type=opener_type,
                context=context
            )
            
            # Call Gemini API
            raw_response = await call_groq_api(prompt, temperature=0.8)
            
            # Parse response
            result = parse_json_response(raw_response)
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=f"Response parsing failed: {result['error']}")
            
            if "reply" not in result:
                raise HTTPException(status_code=500, detail="Missing 'reply' in response")
            
            return {"reply": result["reply"]}
        
        # MODE 3: Curveball Handler
        elif mode == "mode3":
            if not situation_description:
                raise HTTPException(
                    status_code=400,
                    detail="Mode3 requires 'situation_description' field"
                )
            
            # Build prompt
            prompt = build_mode3_prompt(
                situation_description=situation_description,
                mood=mood or "casual",
                image_context=image_context
            )
            
            # Call Gemini API
            raw_response = await call_gemini_api(prompt, temperature=0.8)
            
            # Parse response
            result = parse_json_response(raw_response)
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=f"Response parsing failed: {result['error']}")
            
            if "reply" not in result:
                raise HTTPException(status_code=500, detail="Missing 'reply' in response")
            
            return {"reply": result["reply"]}
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{mode}'. Must be 'mode1', 'mode2', or 'mode3'"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
