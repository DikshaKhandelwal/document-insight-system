import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { selected_text, related_sections, topic, generate_audio } = body
    
    // Forward request to FastAPI backend
    const backendResponse = await fetch('http://localhost:8000/ai-debate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        selected_text,
        related_sections,
        topic: topic || "research_analysis",
        generate_audio: generate_audio !== false // Default to true
      })
    })
    
    if (!backendResponse.ok) {
      throw new Error(`Backend request failed: ${backendResponse.status}`)
    }
    
    const result = await backendResponse.json()
    return NextResponse.json(result)
    
  } catch (error) {
    console.error('AI Debate API error:', error)
    
    // Return enhanced fallback debate for demo purposes
    return NextResponse.json({
      debate_script: [
        {
          speaker: "Dr. Sarah Chen",
          voice: "en-US-JennyNeural",
          text: "I'm concerned about the methodology presented here. How can we verify the statistical significance of these findings without proper peer review?",
          round: 1,
          persona: "skeptic",
          initial: "S"
        },
        {
          speaker: "Prof. Alex Rivera", 
          voice: "en-US-GuyNeural",
          text: "This is incredibly exciting! The implications could revolutionize how we approach this entire field. Think about the practical applications!",
          round: 1,
          persona: "optimist",
          initial: "O"
        },
        {
          speaker: "Dr. Morgan Kim",
          voice: "en-US-AriaNeural", 
          text: "Let's focus on the empirical evidence. The data shows interesting patterns, but we need larger sample sizes before drawing definitive conclusions.",
          round: 1,
          persona: "analyst",
          initial: "A"
        },
        {
          speaker: "Dr. Sarah Chen",
          voice: "en-US-JennyNeural",
          text: "Exactly my point! We're jumping to conclusions without proper controls. Where are the confounding variable analyses?",
          round: 2,
          persona: "skeptic",
          initial: "S"
        },
        {
          speaker: "Prof. Alex Rivera", 
          voice: "en-US-GuyNeural",
          text: "You're both missing the forest for the trees! This could lead to breakthrough innovations we haven't even imagined yet!",
          round: 2,
          persona: "optimist",
          initial: "O"
        },
        {
          speaker: "Dr. Morgan Kim",
          voice: "en-US-AriaNeural", 
          text: "The correlation coefficient suggests a moderate relationship. That's promising, but let's not overstate the findings.",
          round: 2,
          persona: "analyst",
          initial: "A"
        }
      ],
      audio_files: [],
      personas: {
        skeptic: { name: "Dr. Sarah Chen", initial: "S", voice: "en-US-JennyNeural" },
        optimist: { name: "Prof. Alex Rivera", initial: "O", voice: "en-US-GuyNeural" },
        analyst: { name: "Dr. Morgan Kim", initial: "A", voice: "en-US-AriaNeural" }
      },
      topic: "research_analysis",
      participant_count: 3,
      total_segments: 6,
      audio_available: false,
      download_ready: true,
      generation_timestamp: new Date().toISOString(),
      llm_model: "gemini-2.0-flash-exp",
      hackathon_feature: true,
      fallback: true
    })
  }
}
