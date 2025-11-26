from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class ModeType(str, Enum):
    MODE1 = "mode1"
    MODE2 = "mode2"
    MODE3 = "mode3"


class PMood(str, Enum):
    """Personalization moods that affect all conversation responses"""
    PMOOD1 = "pmood1"  # Bold energy
    PMOOD2 = "pmood2"  # Confidence in every word
    PMOOD3 = "pmood3"  # Fun vibes only
    PMOOD4 = "pmood4"  # Soft but strong
    PMOOD5 = "pmood5"  # Flirty vibes


class ChatRequest(BaseModel):
    """
    Universal request schema for all chatbot modes.
    """
    mode_type: ModeType = Field(..., description="The mode of operation")
    
    # Mode 1 fields
    opposite_person_message: Optional[str] = Field(None, description="Other person's message (Mode 1)")
    user_message: Optional[str] = Field(None, description="User's draft reply (Mode 1)")
    mode: Optional[str] = Field(None, description="Tone/mood for Mode 1 or Mode 3")
    
    # Mode 2 fields
    mood: Optional[str] = Field(None, description="Pattern type for Mode 2 or mood for Mode 3")
    
    # Mode 3 fields
    situation_description: Optional[str] = Field(None, description="Description of curveball situation (Mode 3)")
    
    # Common fields
    context: Optional[str] = Field(None, description="Optional context for personalization")
    pmoods: Optional[List[PMood]] = Field(None, description="Personalization moods (pmood1-pmood5). If not provided, all moods apply.")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "mode_type": "mode1",
                    "opposite_person_message": "Hey, what are you up to?",
                    "user_message": "Nothing much",
                    "mode": "casual",
                    "context": "Talking to a friend"
                },
                {
                    "mode_type": "mode2",
                    "mood": "Would you rather...",
                    "context": "Dating app conversation"
                },
                {
                    "mode_type": "mode3",
                    "situation_description": "She just sent three laughing emojis",
                    "mood": "curious"
                }
            ]
        }


class Mode1Response(BaseModel):
    """Response structure for Mode 1"""
    # mode: str
    reply: str


class Mode2Response(BaseModel):
    """Response structure for Mode 2"""
    # mood: str
    # context_used: bool
    reply: str
    # timestamp: datetime


class Mode3Response(BaseModel):
    """Response structure for Mode 3"""
    # mood: str
    reply: str
    # timestamp: datetime


class ChatResponse(BaseModel):
    """
    Universal response schema for all chatbot modes.
    """
    mode_type: ModeType
    mode1_response: Optional[Mode1Response] = None
    mode2_response: Optional[Mode2Response] = None
    mode3_response: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "mode_type": "mode1",
                    "mode1_response": {
        
                        "reply": "Not much! Just chilling. You?",
                        
                    }
                },
                {
                    "mode_type": "mode2",
                    "mode2_response": {
                       
                       
                        "reply": "Would you rather travel back in time or into the future?",
                        
                    }
                },
                {
                    "mode_type": "mode3",
                    "mode3_response": {
                       
                        "reply": "Okay, what's so funny? Did I miss something? 😄",
                        
                    }
                }
            ]
        }
