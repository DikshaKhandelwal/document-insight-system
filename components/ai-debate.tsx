"use client"

import { useState, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  MessageCircle, 
  Brain, 
  Users, 
  Play, 
  Pause, 
  Square,
  Download, 
  RotateCcw, 
  Volume2 
} from "lucide-react"

interface DebateSegment {
  speaker: string
  voice: string
  text: string
  round: number
  persona: string
  initial: string
}

interface AudioFile {
  speaker: string
  filename: string
  text: string
  duration_estimate: number
  persona: string
  round: string | number
  voice: string
}

interface DebateResult {
  debate_script: DebateSegment[]
  audio_files: AudioFile[]
  personas: Record<string, any>
  topic: string
  participant_count: number
  total_segments: number
  audio_available: boolean
  download_ready: boolean
  generation_timestamp: string
  llm_model: string
  hackathon_feature: boolean
}

interface AIDebateProps {
  selectedText: string
  relatedSections: any[]
  onDebateGenerated?: (result: DebateResult) => void
}

const personaIcons = {
  skeptic: "S",
  optimist: "O", 
  analyst: "A"
}

const personaColors = {
  skeptic: "bg-red-100 text-red-700 border-red-200",
  optimist: "bg-green-100 text-green-700 border-green-200",
  analyst: "bg-blue-100 text-blue-700 border-blue-200"
}

export default function AIDebate({ selectedText, relatedSections, onDebateGenerated }: AIDebateProps) {
  const [debate, setDebate] = useState<DebateResult | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentlyPlaying, setCurrentlyPlaying] = useState<number | null>(null)
  const [isPlayingFull, setIsPlayingFull] = useState(false)
  const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null)
  const [debateId, setDebateId] = useState<string>("")
  const [playbackQueue, setPlaybackQueue] = useState<number[]>([])
  const [isPlayingSequence, setIsPlayingSequence] = useState(false)
  const audioRef = useRef<HTMLAudioElement>(null)
  const sequenceRef = useRef<{
    playing: boolean
    currentIndex: number
    totalSegments: number
    segmentAudioFiles: Array<{filename?: string, persona: string, round: number}>
  }>({ 
    playing: false, 
    currentIndex: 0, 
    totalSegments: 0,
    segmentAudioFiles: []
  })

  const generateDebate = async () => {
    if (!selectedText) return
    
    setIsGenerating(true)
    try {
      const response = await fetch('/api/ai-debate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_text: selectedText,
          related_sections: relatedSections,
          topic: "research_analysis",
          generate_audio: true
        })
      })
      
      if (!response.ok) throw new Error('Failed to generate debate')
      
      const result = await response.json()
      setDebate(result)
      setDebateId(Date.now().toString()) // Simple ID for demo
      onDebateGenerated?.(result)
    } catch (error) {
      console.error('Debate generation error:', error)
      // Enhanced fallback debate for demo
      setDebate({
        debate_script: [
          {
            speaker: "Dr. Sarah Chen",
            voice: "en-US-JennyNeural",
            text: "I'm concerned about the methodology here. How can we verify these claims without proper peer review?",
            round: 1,
            persona: "skeptic",
            initial: "S"
          },
          {
            speaker: "Prof. Alex Rivera", 
            voice: "en-US-GuyNeural",
            text: "This is groundbreaking! The potential applications could revolutionize how we approach this field entirely!",
            round: 1,
            persona: "optimist",
            initial: "O"
          },
          {
            speaker: "Dr. Morgan Kim",
            voice: "en-US-AriaNeural", 
            text: "Let's examine the statistical evidence objectively. We need larger sample sizes before drawing definitive conclusions.",
            round: 1,
            persona: "analyst",
            initial: "A"
          },
          {
            speaker: "Dr. Sarah Chen",
            voice: "en-US-JennyNeural",
            text: "Exactly my point! Where are the control groups? What about confounding variables?",
            round: 2,
            persona: "skeptic",
            initial: "S"
          },
          {
            speaker: "Prof. Alex Rivera", 
            voice: "en-US-GuyNeural",
            text: "You're both missing the forest for the trees! This could lead to breakthrough innovations we haven't even imagined!",
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
          skeptic: { name: "Dr. Sarah Chen", initial: "S" },
          optimist: { name: "Prof. Alex Rivera", initial: "O" },
          analyst: { name: "Dr. Morgan Kim", initial: "A" }
        },
        topic: "research_analysis",
        participant_count: 3,
        total_segments: 6,
        audio_available: false,
        download_ready: true,
        generation_timestamp: new Date().toISOString(),
        llm_model: "gemini-2.0-flash-exp",
        hackathon_feature: true
      })
      setDebateId(Date.now().toString())
    } finally {
      setIsGenerating(false)
    }
  }

  const stopCurrentAudio = () => {
    console.log('Stopping current audio')
    if (currentAudio) {
      currentAudio.pause()
      currentAudio.currentTime = 0
      setCurrentAudio(null)
    }
    setCurrentlyPlaying(null)
    setIsPlayingFull(false)
    setIsPlayingSequence(false)
    setPlaybackQueue([])
    
    // Stop sequence playback
    sequenceRef.current.playing = false
  }

  const playSegment = async (index: number, audioFilename?: string) => {
    // If currently playing the same segment, stop it
    if (currentlyPlaying === index && currentAudio) {
      stopCurrentAudio()
      return
    }
    
    // Stop any currently playing audio or sequence
    stopCurrentAudio()
    
    setCurrentlyPlaying(index)
    
    try {
      let audioUrl: string
      
      if (audioFilename && debate?.audio_available) {
        // Play generated audio file
        audioUrl = `/api/audio/${audioFilename}`
      } else {
        // Generate TTS for the segment
        const segment = debate?.debate_script[index]
        if (!segment) return
        
        const response = await fetch('/api/audio', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            selected_text: segment.text,
            related_sections: [],
            insights: {},
            script_style: "single_voice",
            voice_name: segment.voice
          })
        })
        
        if (!response.ok) {
          console.error('TTS generation failed')
          setCurrentlyPlaying(null)
          return
        }
        
        const result = await response.json()
        if (result.audio_files && result.audio_files.length > 0) {
          audioUrl = `/api/audio/${result.audio_files[0].filename}`
        } else {
          throw new Error('No audio file generated')
        }
      }
      
      const audio = new Audio(audioUrl)
      setCurrentAudio(audio)
      
      audio.onended = () => {
        setCurrentlyPlaying(null)
        setCurrentAudio(null)
      }
      
      audio.onerror = () => {
        console.error('Audio playback failed')
        setCurrentlyPlaying(null)
        setCurrentAudio(null)
      }
      
      await audio.play()
      
    } catch (error) {
      console.error('Audio playback error:', error)
      setCurrentlyPlaying(null)
    }
  }

  const playAllSegments = () => {
    if (!debate?.debate_script || debate.debate_script.length === 0) return
    
    console.log('Starting Play All - Total segments:', debate.debate_script.length)
    
    // If currently playing sequence, stop it
    if (sequenceRef.current.playing) {
      console.log('Stopping current sequence')
      stopCurrentAudio()
      return
    }
    
    // Stop any current audio
    stopCurrentAudio()
    
    // Set up sequence info
    sequenceRef.current = {
      playing: true,
      currentIndex: 0,
      totalSegments: debate.debate_script.length,
      segmentAudioFiles: debate.debate_script.map(segment => {
        const audioFile = debate.audio_files?.find(f => 
          f.persona === segment.persona && f.round === segment.round
        )
        return {
          filename: audioFile?.filename,
          persona: segment.persona,
          round: segment.round
        }
      })
    }
    
    setIsPlayingSequence(true)
    setIsPlayingFull(true)
    
    console.log('Starting sequence playback')
    playSequenceSegment(0)
  }

  const playSequenceSegment = async (index: number) => {
    if (!debate?.debate_script || !sequenceRef.current.playing || index >= sequenceRef.current.totalSegments) {
      console.log('Sequence ended or stopped')
      sequenceRef.current.playing = false
      setIsPlayingSequence(false)
      setIsPlayingFull(false)
      return
    }

    console.log(`Playing sequence segment ${index + 1}/${sequenceRef.current.totalSegments}`)
    sequenceRef.current.currentIndex = index
    
    const segment = debate.debate_script[index]
    const audioFile = sequenceRef.current.segmentAudioFiles[index]
    
    setCurrentlyPlaying(index)
    
    try {
      let audioUrl: string
      
      if (audioFile.filename && debate.audio_available) {
        audioUrl = `/api/audio/${audioFile.filename}`
        console.log('Using pre-generated audio:', audioFile.filename)
      } else {
        console.log('Generating TTS for segment:', index)
        const response = await fetch('/api/audio', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            selected_text: segment.text,
            related_sections: [],
            insights: {},
            script_style: "single_voice",
            voice_name: segment.voice
          })
        })
        
        if (!response.ok) {
          console.error('TTS generation failed for segment:', index)
          // Try next segment
          setTimeout(() => playSequenceSegment(index + 1), 1000)
          return
        }
        
        const result = await response.json()
        if (result.audio_files && result.audio_files.length > 0) {
          audioUrl = `/api/audio/${result.audio_files[0].filename}`
        } else {
          console.error('No audio file generated for segment:', index)
          setTimeout(() => playSequenceSegment(index + 1), 1000)
          return
        }
      }
      
      const audio = new Audio(audioUrl)
      setCurrentAudio(audio)
      
      audio.onended = () => {
        console.log('Segment', index, 'ended, playing next')
        setCurrentlyPlaying(null)
        setCurrentAudio(null)
        
        if (sequenceRef.current.playing) {
          setTimeout(() => {
            playSequenceSegment(index + 1)
          }, 500)
        }
      }
      
      audio.onerror = () => {
        console.error('Audio playback failed for segment:', index)
        setCurrentlyPlaying(null)
        setCurrentAudio(null)
        
        if (sequenceRef.current.playing) {
          setTimeout(() => {
            playSequenceSegment(index + 1)
          }, 1000)
        }
      }
      
      await audio.play()
      console.log('Started playing segment:', index)
      
    } catch (error) {
      console.error('Error playing segment', index, ':', error)
      setCurrentlyPlaying(null)
      
      if (sequenceRef.current.playing) {
        setTimeout(() => {
          playSequenceSegment(index + 1)
        }, 1000)
      }
    }
  }

  const playFullDebate = async () => {
    if (!debate) return
    
    stopCurrentAudio()
    setIsPlayingFull(true)
    
    try {
      // Check if we have a combined audio file
      const combinedAudio = debate.audio_files?.find(f => f.persona === 'combined')
      
      if (combinedAudio) {
        // Play the combined debate audio
        const audioUrl = `/api/audio/${combinedAudio.filename}`
        const audio = new Audio(audioUrl)
        setCurrentAudio(audio)
        
        audio.onended = () => {
          setIsPlayingFull(false)
          setCurrentAudio(null)
        }
        
        audio.onerror = () => {
          console.error('Full debate audio playback failed')
          setIsPlayingFull(false)
          setCurrentAudio(null)
        }
        
        await audio.play()
      } else {
        // Play segments sequentially
        for (let i = 0; i < debate.debate_script.length; i++) {
          if (!isPlayingFull) break // User stopped playback
          
          await playSegment(i)
          
          // Wait for segment to finish before playing next
          await new Promise((resolve) => {
            const checkAudio = setInterval(() => {
              if (!currentAudio || currentAudio.ended || currentAudio.paused) {
                clearInterval(checkAudio)
                resolve(void 0)
              }
            }, 100)
          })
          
          // Small pause between speakers
          await new Promise(resolve => setTimeout(resolve, 800))
        }
        
        setIsPlayingFull(false)
      }
    } catch (error) {
      console.error('Full debate playback error:', error)
      setIsPlayingFull(false)
    }
  }

  const downloadTranscript = async () => {
    if (!debateId) return
    
    try {
      const response = await fetch(`/api/ai-debate/download/${debateId}`)
      if (!response.ok) throw new Error('Download failed')
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `ai_debate_transcript_${debateId}.txt`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Download error:', error)
      
      // Fallback: generate and download transcript locally
      if (debate) {
        const transcript = generateTranscriptText(debate)
        const blob = new Blob([transcript], { type: 'text/plain' })
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `ai_debate_transcript_${debateId}.txt`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        window.URL.revokeObjectURL(url)
      }
    }
  }

  const generateTranscriptText = (debate: DebateResult): string => {
    const timestamp = new Date().toLocaleString()
    
    let transcript = `AI EXPERT DEBATE TRANSCRIPT\n`
    transcript += `Generated: ${timestamp}\n`
    transcript += `Debate ID: ${debateId}\n`
    transcript += `Model: ${debate.llm_model || 'Gemini 2.5 Flash'}\n\n`
    
    transcript += `=================================================\n`
    transcript += `PARTICIPANTS:\n`
    transcript += `[S] Dr. Sarah Chen - The Skeptical Researcher\n`
    transcript += `[O] Prof. Alex Rivera - The Optimistic Innovator\n`
    transcript += `[A] Dr. Morgan Kim - The Data-Driven Analyst\n`
    transcript += `=================================================\n\n`
    
    // Group by rounds
    const rounds = new Set(debate.debate_script.map(s => s.round))
    
    rounds.forEach(round => {
      transcript += `ROUND ${round}: ${round === 1 ? 'INITIAL PERSPECTIVES' : 'RESPONSES & REBUTTALS'}\n`
      
      debate.debate_script
        .filter(s => s.round === round)
        .forEach(segment => {
          transcript += `[${segment.speaker}]: ${segment.text}\n\n`
        })
    })
    
    transcript += `=================================================\n`
    transcript += `GENERATED BY: Document Insight System\n`
    transcript += `POWERED BY: Gemini 2.5 Flash + Azure Speech Services\n`
    transcript += `=================================================\n`
    
    return transcript
  }

  return (
    <div className="w-full">
      <Card className="w-full border-0 shadow-none">
        <CardHeader className="pb-2 px-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Users className="h-4 w-4" />
              AI Expert Debate
            </CardTitle>
            <Badge variant="secondary" className="bg-gradient-to-r from-purple-500 to-pink-500 text-white text-xs px-1 py-0">
              EXCLUSIVE
            </Badge>
          </div>
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground">
              Three AI personalities debate your research
            </p>
            {debate && (
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <span className="text-xs">Powered by {debate.llm_model}</span>
                {debate.audio_available && <Volume2 className="h-3 w-3" />}
              </div>
            )}
          </div>
        </CardHeader>
        
        <CardContent className="space-y-3 px-3 pb-3">
          {!debate ? (
            <div className="text-center py-4">
              <Brain className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
              <h3 className="text-sm font-medium mb-1">Ready for an AI Debate?</h3>
              <p className="text-xs text-muted-foreground mb-3">
                AI experts will analyze your selected text using Gemini 2.5 Flash
              </p>
              <Button 
                onClick={generateDebate} 
                disabled={!selectedText || isGenerating}
                className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
                size="sm"
              >
                {isGenerating ? (
                  <>
                    <Brain className="mr-1 h-3 w-3 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <MessageCircle className="mr-1 h-3 w-3" />
                    Start Debate
                  </>
                )}
              </Button>
            </div>
          ) : (
            <div className="space-y-2">
              {/* Debate Controls */}
              <div className="flex items-center justify-between border-b pb-1">
                <div className="flex items-center gap-1">
                  <h3 className="font-medium text-xs">AI Panel</h3>
                  <Badge variant="outline" className="text-xs px-1">{debate.participant_count}</Badge>
                  {debate.audio_available && (
                    <Badge variant="outline" className="bg-green-50 text-green-700 text-xs px-1">
                      <Volume2 className="h-2 w-2 mr-1" />
                      Audio
                    </Badge>
                  )}
                </div>
                <div className="flex items-center gap-1">
                  <Button 
                    onClick={isPlayingSequence ? stopCurrentAudio : playAllSegments}
                    variant="outline"
                    size="sm"
                    className="text-xs px-1 py-0 h-6"
                  >
                    {isPlayingSequence ? (
                      <>
                        <Square className="h-2 w-2 mr-1" />
                        Stop All
                      </>
                    ) : (
                      <>
                        <Play className="h-2 w-2 mr-1" />
                        Play All
                      </>
                    )}
                  </Button>
                  <Button 
                    onClick={downloadTranscript}
                    variant="outline"
                    size="sm"
                    className="text-xs px-1 py-0 h-6"
                  >
                    <Download className="h-2 w-2" />
                  </Button>
                </div>
              </div>

              {/* Compact Personas Legend */}
              <div className="flex gap-1 text-xs">
                <div className="flex items-center gap-1 p-1 bg-red-50 rounded text-xs">
                  <div className="w-4 h-4 rounded-full bg-red-200 text-red-700 flex items-center justify-center font-semibold text-xs">
                    S
                  </div>
                  <span className="text-xs">Skeptic</span>
                </div>
                <div className="flex items-center gap-1 p-1 bg-green-50 rounded text-xs">
                  <div className="w-4 h-4 rounded-full bg-green-200 text-green-700 flex items-center justify-center font-semibold text-xs">
                    O
                  </div>
                  <span className="text-xs">Optimist</span>
                </div>
                <div className="flex items-center gap-1 p-1 bg-blue-50 rounded text-xs">
                  <div className="w-4 h-4 rounded-full bg-blue-200 text-blue-700 flex items-center justify-center font-semibold text-xs">
                    A
                  </div>
                  <span className="text-xs">Analyst</span>
                </div>
              </div>

              {/* Compact Debate Script */}
              <div className="max-h-48 overflow-y-auto border rounded p-2 bg-gray-50/30">
                <div className="space-y-1">
                  {debate.debate_script.map((segment, index) => {
                    const audioFile = debate.audio_files?.find(f => 
                      f.persona === segment.persona && f.round === segment.round
                    )
                    
                    return (
                      <div key={index} className="flex gap-1 p-1 bg-white rounded text-xs">
                        <div className="flex-shrink-0">
                          <div className={`
                            w-5 h-5 rounded-full flex items-center justify-center text-xs font-semibold
                            ${personaColors[segment.persona as keyof typeof personaColors]}
                          `}>
                            {segment.initial || personaIcons[segment.persona as keyof typeof personaIcons]}
                          </div>
                        </div>
                        
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1 mb-0.5">
                            <span className="font-medium text-xs truncate">{segment.speaker.split(' ')[2]}</span>
                            <Badge variant="outline" className="text-xs px-1">R{segment.round}</Badge>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => playSegment(index, audioFile?.filename)}
                              className="h-4 w-4 p-0 flex-shrink-0"
                            >
                              {currentlyPlaying === index ? (
                                <Square className="h-2 w-2 text-red-600" />
                              ) : (
                                <Play className="h-2 w-2" />
                              )}
                            </Button>
                            {audioFile && (
                              <span className="text-xs text-muted-foreground flex-shrink-0">
                                {audioFile.duration_estimate}s
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground leading-relaxed break-words">
                            {segment.text}
                          </p>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Compact Action Buttons */}
              <div className="flex gap-1 pt-1 border-t">
                <Button 
                  onClick={generateDebate} 
                  variant="outline" 
                  className="flex-1 text-xs h-6"
                  disabled={isGenerating}
                >
                  {isGenerating ? (
                    <>
                      <Brain className="mr-1 h-2 w-2 animate-spin" />
                      Regenerating...
                    </>
                  ) : (
                    <>
                      <RotateCcw className="mr-1 h-2 w-2" />
                      New Debate
                    </>
                  )}
                </Button>
              </div>
              
              {/* Compact Debug Info */}
              {debate.generation_timestamp && (
                <div className="text-xs text-muted-foreground text-center">
                  {new Date(debate.generation_timestamp).toLocaleTimeString()}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
