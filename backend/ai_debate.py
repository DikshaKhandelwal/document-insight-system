# AI Debate Feature - Hackathon MVP
# Add this to your main.py or import as module

from pydantic import BaseModel
from typing import List, Dict, Any
import google.generativeai as genai

class DebateRequest(BaseModel):
    selected_text: str
    related_sections: List[Dict[str, Any]]
    topic: str = "research_analysis"

async def generate_ai_debate(request: DebateRequest):
    """Generate a 3-way AI personality debate"""
    
    # Define 3 distinct AI personas
    personas = {
        "skeptic": {
            "name": "Dr. Sarah Chen",
            "voice": "en-US-JennyNeural",  # Professional female voice
            "personality": "You are Dr. Sarah Chen, a skeptical academic who questions methodology, looks for flaws, and demands rigorous evidence. Always ask 'But what about...?' and point out limitations.",
            "style": "questioning, precise, critical"
        },
        "optimist": {
            "name": "Prof. Alex Rivera", 
            "voice": "en-US-GuyNeural",  # Enthusiastic male voice
            "personality": "You are Prof. Alex Rivera, an enthusiastic researcher who sees potential and applications everywhere. You build on ideas and find positive connections.",
            "style": "enthusiastic, forward-thinking, collaborative"
        },
        "analyst": {
            "name": "Dr. Morgan Kim",
            "voice": "en-US-AriaNeural",  # Neutral analytical voice  
            "personality": "You are Dr. Morgan Kim, a data-driven analyst who focuses purely on evidence, statistics, and logical conclusions. You moderate between other viewpoints.",
            "style": "logical, evidence-based, balanced"
        }
    }
    
    # Generate debate script
    debate_script = []
    
    # Round 1: Initial positions
    for persona_key, persona in personas.items():
        prompt = f"""
        {persona['personality']}
        
        Topic: "{request.selected_text}"
        
        Related research context:
        {format_related_sections(request.related_sections)}
        
        Give your initial perspective on this topic in 2-3 sentences. 
        Stay true to your personality - be {persona['style']}.
        """
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            debate_script.append({
                "speaker": persona["name"],
                "voice": persona["voice"],
                "text": response.text.strip(),
                "round": 1,
                "persona": persona_key
            })
        except Exception as e:
            # Fallback responses
            fallbacks = {
                "skeptic": f"I'm concerned about the methodology here. How can we be sure that {request.selected_text[:50]}... is actually valid given the limited sample size?",
                "optimist": f"This is exciting! The implications of {request.selected_text[:50]}... could revolutionize how we approach this field entirely!",
                "analyst": f"Looking at the data from {len(request.related_sections)} related studies, we need to examine the statistical significance before drawing conclusions."
            }
            debate_script.append({
                "speaker": persona["name"],
                "voice": persona["voice"], 
                "text": fallbacks[persona_key],
                "round": 1,
                "persona": persona_key
            })
    
    # Round 2: Responses and rebuttals
    context = "\n".join([f"{item['speaker']}: {item['text']}" for item in debate_script])
    
    for persona_key, persona in personas.items():
        prompt = f"""
        {persona['personality']}
        
        Previous debate points:
        {context}
        
        Now respond to the other perspectives. Challenge or build upon their points.
        Keep it to 2-3 sentences and maintain your {persona['style']} approach.
        """
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            debate_script.append({
                "speaker": persona["name"],
                "voice": persona["voice"],
                "text": response.text.strip(),
                "round": 2, 
                "persona": persona_key
            })
        except Exception as e:
            # Fallback rebuttals
            fallbacks = {
                "skeptic": "Hold on - I think we're getting ahead of ourselves. Where's the peer review? What about confounding variables?",
                "optimist": "You're both missing the bigger picture! This could lead to breakthrough applications we haven't even considered yet!",
                "analyst": "Let's focus on the numbers. The correlation coefficient suggests a moderate relationship, nothing more."
            }
            debate_script.append({
                "speaker": persona["name"],
                "voice": persona["voice"],
                "text": fallbacks[persona_key],
                "round": 2,
                "persona": persona_key
            })
    
    return {
        "debate_script": debate_script,
        "personas": personas,
        "topic": request.topic,
        "participant_count": 3,
        "total_segments": len(debate_script)
    }

def format_related_sections(sections):
    """Format related sections for AI context"""
    formatted = []
    for section in sections[:3]:  # Limit to top 3 for brevity
        formatted.append(f"- {section.get('document_name', 'Unknown')}: {section.get('snippet', '')[:100]}...")
    return "\n".join(formatted)

# Add to main.py:
@app.post("/ai-debate")
async def create_ai_debate(request: DebateRequest):
    """Generate an AI personality debate about the selected content"""
    try:
        result = await generate_ai_debate(request)
        return result
    except Exception as e:
        return {
            "error": f"Debate generation failed: {str(e)}",
            "fallback_script": [
                {"speaker": "Dr. Sarah Chen", "text": "I have concerns about this research approach.", "persona": "skeptic"},
                {"speaker": "Prof. Alex Rivera", "text": "But think about the potential applications!", "persona": "optimist"}, 
                {"speaker": "Dr. Morgan Kim", "text": "Let's examine the data objectively.", "persona": "analyst"}
            ]
        }
