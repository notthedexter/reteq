import os
import json
import logging
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from groq import Groq
from app.services.chatbot.chatbot_schema import ChatRequest, ChatResponse, ModeType, Mode1Response, Mode2Response
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()
logger = logging.getLogger(__name__)

class ChatbotService:
    
    PMOOD_DEFINITIONS = {
        "pmood1": """Bold Energy: Make every word count with impact and strength. Use assertive language, 
        strong statements, and confident phrasing. Your message should command attention and leave no room 
        for ambiguity. Be direct, powerful, and unapologetic in your expression.""",
        
        "pmood2": """Confidence in Every Word: Express yourself with unwavering self-assurance and poise. 
        Use language that demonstrates certainty, competence, and ease. Avoid hedging words like 'maybe', 
        'I think', or 'perhaps'. Instead, state things with conviction and natural authority.""",
        
        "pmood3": """Fun Vibes Only: Keep it light, playful, and effortlessly entertaining. Use humor, 
        wit, and upbeat energy. Include playful language, casual expressions, and an easy-going tone 
        that makes the conversation feel enjoyable and stress-free. Make them smile.""",
        
        "pmood4": """Soft but Strong: Balance warmth with confidence. Be kind, empathetic, and genuine 
        while maintaining your ground. Use gentle language that shows care and understanding, but never 
        comes across as weak or uncertain. Demonstrate emotional intelligence and authentic connection.""",
        
        "pmood5": """Flirty Vibes: Add subtle playfulness and romantic intrigue. Use light teasing, 
        playful banter, and charming language that creates attraction. Be engaging and slightly mysterious, 
        with hints of interest that invite reciprocation. Keep it classy, not overly forward."""
    }
    
    MODE1_TONES = {
        "casual": "Keep the tone light, friendly, and conversational. Use casual language, humor, and a relaxed vibe.",
        "flirty": "Add a playful, charming tone with light teasing. Be engaging, witty, and slightly flirty while remaining respectful.",
        "nonchalant": "Maintain a cool, unbothered attitude. Be witty, slightly detached, and confident while staying genuine."
    }
    
    MODE2_PATTERNS = [
        "If you...",
        "Between...",
        "If I was...",
        "Would you rather...",
        "What is something that...",
        "In a situation where...",
        "How would you handle..."
    ]
    
    MODE3_MOODS = {
        "casual": "Keep it light, relaxed, and easy-going. Don't overthink the situation.",
        "apologetic": "Be understanding and apologetic. Show you care about the confusion or misunderstanding.",
        "encouraging": "Be positive, supportive, and uplifting. Turn the awkward moment into something good.",
        "sarcastic": "Use witty sarcasm and humor to deflect or address the situation playfully (not mean).",
        "curious": "Show genuine interest and curiosity. Ask questions to understand better.",
        "flirty": "Turn the awkward moment into a playful, flirty opportunity while staying smooth.",
        "empathetic": "Show deep understanding and emotional connection. Validate their feelings or situation."
    }
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY not set in environment variables")
            raise ValueError("GROQ_API_KEY is required")
        
        self.client = Groq(api_key=api_key)
        # Available Groq models: llama-3.3-70b-versatile, llama-3.1-70b-versatile, mixtral-8x7b-32768
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        logger.info(f"Initialized Groq client with model: {self.model}")

    def _build_pmood_instruction(self, pmoods_off: Optional[List[str]] = None) -> str:
        """
        Build instruction text for personalization moods.
        All moods are ON by default. Pass pmoods_off list to disable specific moods.
        """
        # Start with all moods enabled
        all_moods = list(self.PMOOD_DEFINITIONS.keys())
        
        # Remove disabled moods
        if pmoods_off and len(pmoods_off) > 0:
            active_moods = [mood for mood in all_moods if mood not in pmoods_off]
        else:
            # All moods are active by default
            active_moods = all_moods
        
        if not active_moods:
            return ""
        
        mood_descriptions = [self.PMOOD_DEFINITIONS.get(mood, "") for mood in active_moods if mood in self.PMOOD_DEFINITIONS]
        
        if not mood_descriptions:
            return ""
        
        instruction = "\n\nPERSONALIZATION LAYER - Apply these personality traits to your response:\n"
        instruction += "\n".join([f"{desc}" for desc in mood_descriptions])
        instruction += "\n\nBlend these traits naturally into your response without being heavy-handed."
        
        return instruction

    def _call_groq(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Call Groq API with chat completion format
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise

    def analyze_mode1_context(self, opposite_message: str, user_message: str, mode: Optional[str] = None, context: Optional[str] = None, pmoods_off: Optional[List[str]] = None) -> Dict:
        """
        First-pass analysis to understand the conversation and determine tone approach.
        """
        history_context = f"Other person: {opposite_message}\nUser wants to say: {user_message}"
        
        system_prompt = """You are a conversation tone analyzer. Analyze the conversation and determine:
1. The emotional context and dynamics
2. Whether the mode/context fits the conversation naturally
3. What tone would work best

Respond in JSON format with analysis."""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": history_context}
            ]
            result_text = self._call_groq(messages, temperature=0.5, max_tokens=150)
            
            # Try to parse JSON
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                # Extract JSON if embedded in text
                start = result_text.find("{")
                end = result_text.rfind("}")
                if start != -1 and end != -1:
                    result = json.loads(result_text[start:end+1])
                    return result
                return {"analysis": "Conversation context noted"}
        except Exception as e:
            logger.warning(f"Error in tone analysis: {e}")
            return {"analysis": "Conversation context noted"}

    def generate_mode1_reply(self, opposite_message: str, user_message: str, mode: Optional[str] = None, context: Optional[str] = None, pmoods_off: Optional[List[str]] = None) -> str:
        """
        Generate a contextual reply based on the conversation and specified tone.
        """
        if context and len(context.strip()) > 0:
            tone_instruction = f"Shape your response based on this context: {context}"
        else:
            mode_name = mode if mode else "casual"
            tone_instruction = self.MODE1_TONES.get(mode_name, self.MODE1_TONES["casual"])
        
        pmood_instruction = self._build_pmood_instruction(pmoods_off)
        
        system_prompt = """You are an expert conversationalist skilled in crafting witty, engaging replies.

TASK:
- You have another person's message and what the user wants to say
- Generate a natural, engaging reply that incorporates the user's message
- The reply should respond directly to the other person's message
- Shape the tone/style according to the specified instructions
- Apply the Personalization Layer to set the overall vibe of your response
- Keep it authentic and conversational (1-3 sentences typically)

RESPONSE FORMAT:
Provide ONLY the reply message, nothing else. No JSON, no explanations."""
        
        user_prompt = f"""
OTHER PERSON'S MESSAGE:
"{opposite_message}"

WHAT I WANT TO SAY:
"{user_message}"

TONE INSTRUCTION:
{tone_instruction}{pmood_instruction}

Generate a natural reply that incorporates what I want to say while responding to their message in the specified tone."""
        
        try:
            logger.info(f"Generating Mode 1 reply with tone: {mode}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            reply = self._call_groq(messages, temperature=0.8, max_tokens=200)
            logger.info("Successfully generated Mode 1 reply")
            return reply
        except Exception as e:
            logger.error(f"Error generating Mode 1 reply: {e}")
            return f"That's cool! {user_message}"

    def analyze_mode2_intent(self, mood: Optional[str] = None, context: Optional[str] = None, pmoods_off: Optional[List[str]] = None) -> Dict:
        """
        Analyze the intent for generating a conversation opener.
        """
        if context and len(context.strip()) > 0:
            intent = f"Generate an opener based on context: {context}"
        else:
            mood_pattern = mood if mood else self.MODE2_PATTERNS[0]
            intent = f"Generate an opener using pattern: {mood_pattern}"
        
        system_prompt = """You are an icebreaker expert. Analyze the requested opener type and return JSON with:
1. pattern_type: The mood pattern to use
2. contextual_hints: Key themes to weave in
3. tone: Suggested tone for the opener"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": intent}
            ]
            result_text = self._call_groq(messages, temperature=0.5, max_tokens=120)
            
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                start = result_text.find("{")
                end = result_text.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(result_text[start:end+1])
                return {"pattern_type": mood or self.MODE2_PATTERNS[0]}
        except Exception as e:
            logger.warning(f"Error in intent analysis: {e}")
            return {"pattern_type": mood or self.MODE2_PATTERNS[0]}

    def generate_mode2_opener(self, mood: Optional[str] = None, context: Optional[str] = None, pmoods_off: Optional[List[str]] = None) -> str:
        """
        Generate a conversation opener based on mood pattern and optional context.
        """
        if mood and mood not in self.MODE2_PATTERNS:
            mood = self.MODE2_PATTERNS[0]
        
        mood = mood or self.MODE2_PATTERNS[0]
        pmood_instruction = self._build_pmood_instruction(pmoods_off)
        
        system_prompt = """You are an expert at creating engaging conversation starters.

TASK:
- Generate ONE compelling conversation opener using the specified mood pattern
- The opener should be natural, thought-provoking, and easy to respond to
- Match the specified pattern structure exactly
- If context is provided, weave it naturally into the opener
- Apply the Personalization Layer to set the overall vibe of your opener

RESPONSE FORMAT:
Provide ONLY the conversation opening statement, nothing else. No JSON, no explanations.
Keep it concise (1-2 sentences maximum)."""
        
        if context and len(context.strip()) > 0:
            user_prompt = f"""
MOOD PATTERN: {mood}
CONTEXT: {context}{pmood_instruction}

Generate a single, engaging conversation opener that starts with "{mood}" and naturally incorporates the provided context."""
        else:
            user_prompt = f"""
MOOD PATTERN: {mood}{pmood_instruction}

Generate a single, engaging conversation opener that starts with "{mood}" and is universally relatable."""
        
        try:
            logger.info(f"Generating Mode 2 opener with pattern: {mood}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            opener = self._call_groq(messages, temperature=0.75, max_tokens=120)
            logger.info("Successfully generated Mode 2 opener")
            return opener
        except Exception as e:
            logger.error(f"Error generating Mode 2 opener: {e}")
            return f"{mood} something interesting?"

    def generate_mode3_curveball(self, situation: str, mood: Optional[str] = None, pmoods_off: Optional[List[str]] = None) -> str:
        """
        Generate a reply to handle curveball/awkward situations.
        """
        if mood and mood not in self.MODE3_MOODS:
            mood = "casual"
        
        mood = mood or "casual"
        tone_instruction = self.MODE3_MOODS.get(mood, self.MODE3_MOODS["casual"])
        pmood_instruction = self._build_pmood_instruction(pmoods_off)
        
        system_prompt = """You are an expert at handling awkward, confusing, or curveball moments in conversations.

TASK:
- Analyze the curveball/awkward situation described
- Generate a reply that handles it according to the specified mood
- Make it natural, authentic, and conversation-appropriate
- Keep it concise (1-3 sentences typically)
- Turn the awkward moment into a smooth continuation of the conversation
- Apply the Personalization Layer to set the overall vibe of your response

RESPONSE FORMAT:
Provide ONLY the reply message, nothing else. No JSON, no explanations."""
        
        user_prompt = f"""
SITUATION:
"{situation}"

TARGET MOOD:
{tone_instruction}{pmood_instruction}

Generate a smooth reply that handles this curveball situation in the specified mood."""
        
        try:
            logger.info(f"Generating Mode 3 curveball reply with mood: {mood}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            reply = self._call_groq(messages, temperature=0.8, max_tokens=200)
            logger.info("Successfully generated Mode 3 curveball reply")
            return reply
        except Exception as e:
            logger.error(f"Error generating Mode 3 reply: {e}")
            return "I'm not quite sure how to respond to that, but let's keep talking!"

    async def get_response(self, request: ChatRequest) -> ChatResponse:
        """
        Main entry point for processing chat requests
        """
        try:
            if request.mode_type == ModeType.MODE1:
                if not request.opposite_person_message or not request.user_message:
                    raise ValueError("Mode 1 requires opposite_person_message and user_message")
                
                # Analyze context
                analysis = self.analyze_mode1_context(
                    request.opposite_person_message,
                    request.user_message,
                    request.mode,
                    request.context,
                    request.pmoods if hasattr(request, 'pmoods') else None
                )
                
                # Generate reply
                reply = self.generate_mode1_reply(
                    request.opposite_person_message,
                    request.user_message,
                    request.mode,
                    request.context,
                    request.pmoods if hasattr(request, 'pmoods') else None
                )
                
                logger.info(f"Processed Mode 1 request successfully")
                return ChatResponse(
                    mode_type=ModeType.MODE1,
                    mode1_response=Mode1Response(
                        mode=request.mode or "casual",
                        reply=reply,
                        timestamp=datetime.now()
                    )
                )
            
            elif request.mode_type == ModeType.MODE2:
                # Analyze intent
                analysis = self.analyze_mode2_intent(request.mood, request.context, request.pmoods if hasattr(request, 'pmoods') else None)
                
                # Generate opener
                opener = self.generate_mode2_opener(request.mood, request.context, request.pmoods if hasattr(request, 'pmoods') else None)
                
                logger.info(f"Processed Mode 2 request successfully")
                return ChatResponse(
                    mode_type=ModeType.MODE2,
                    mode2_response=Mode2Response(
                        mood=request.mood or self.MODE2_PATTERNS[0],
                        context_used=bool(request.context and len(request.context.strip()) > 0),
                        opener=opener,
                        timestamp=datetime.now()
                    )
                )
            
            elif request.mode_type == ModeType.MODE3:
                if not request.situation_description:
                    raise ValueError("Mode 3 requires situation_description")
                
                # Generate curveball reply
                reply = self.generate_mode3_curveball(
                    request.situation_description,
                    request.mood,
                    request.pmoods if hasattr(request, 'pmoods') else None
                )
                
                logger.info(f"Processed Mode 3 request successfully")
                return ChatResponse(
                    mode_type=ModeType.MODE3,
                    mode3_response={
                        "mood": request.mood or "casual",
                        "curveball_reply": reply,
                        "timestamp": datetime.now()
                    }
                )
            else:
                raise ValueError(f"Invalid mode type: {request.mode_type}")
                
        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            raise

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def root():
    return {
        "message": "SocialWiz AI - Your Conversation Assistant",
        "version": "2.0.0",
        "modes": {
            "mode1": {
                "name": "Message Rewriter",
                "endpoint": "/api/mode1/rewrite",
                "description": "Rewrite messages with different moods (casual, flirty, nonchalant, slap_back)"
            },
            "mode2": {
                "name": "Icebreaker Generator",
                "endpoint": "/api/mode2/generate",
                "description": "Generate conversation openers with various patterns"
            },
            "mode3": {
                "name": "Curveball Handler",
                "endpoint": "/api/mode3/handle",
                "description": "Handle awkward situations with different moods (casual, apologetic, encouraging, sarcastic, curious, flirty, empathetic)"
            }
        },
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SocialWiz AI Chatbot",
        "modes_active": ["mode1", "mode2", "mode3"]
    }