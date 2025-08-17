"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Search, FileText, Brain, Volume2, ArrowLeft, Sparkles, Lightbulb, Target, Upload, Loader2, Play, Pause, Volume1, Download, AlertCircle, Eye, ExternalLink } from "lucide-react"
import Link from "next/link"
import { Alert, AlertDescription } from "@/components/ui/alert"
import PDFViewer from "@/components/pdf-viewer"
import AudioPlayer from "@/components/audio-player"

interface SearchResult {
  id: string
  document_name: string
  section_title: string
  snippet: string
  similarity_score: number
  page_number: number
  highlight_text: string
}

interface InsightData {
  related?: string
  overlapping?: string
  contradicting?: string
  examples?: string
  extensions?: string
  problems?: string
  applications?: string
  methodology?: string
  status?: string
  summary?: string
  key_points?: string[]
  recommendations?: string[]
}

interface AudioSegment {
  speaker: string
  filename: string
  text: string
  duration_estimate?: number
}

// Enhanced PDF.js integration
interface PDFViewerRef {
  jumpToPage: (page: number, x?: number, y?: number) => void
  highlightText: (text: string, page: number) => void
  getSelectedText: () => string
  clearSelection: (page: number) => void
  enableSelection: (enabled: boolean) => void
}

export default function ReaderPage() {
  const [selectedText, setSelectedText] = useState("")
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [insights, setInsights] = useState<InsightData>({})
  const [audioSegments, setAudioSegments] = useState<AudioSegment[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [isGeneratingInsights, setIsGeneratingInsights] = useState(false)
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false)
  const [currentPdf, setCurrentPdf] = useState<string | null>(null)
  const [documents, setDocuments] = useState<any[]>([])
  const [uploadProgress, setUploadProgress] = useState<number>(0)
  const [isUploading, setIsUploading] = useState(false)
  const [currentAudioIndex, setCurrentAudioIndex] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [activeTab, setActiveTab] = useState("related")
  
  const pdfViewerRef = useRef<PDFViewerRef>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  // Force audio element to remount when a new file arrives to avoid stale metadata/state
  const audioKey = audioSegments[0]?.filename || 'audio'

  useEffect(() => {
    loadDocuments()
  }, [])

  const loadDocuments = async () => {
    try {
      const response = await fetch('/api/documents')
      if (response.ok) {
        const docs = await response.json()
        setDocuments(docs)
        
        // If no current PDF is loaded and we have documents, load the first one
        if (!currentPdf && docs.length > 0) {
          const firstDoc = docs[0]
          if (firstDoc.filename) {
            const timestamp = Date.now()
            const pdfUrl = `/api/files/${firstDoc.filename}?t=${timestamp}&cache=false`
            console.log('Auto-loading first document:', firstDoc.filename)
            setCurrentPdf(pdfUrl)
          }
        }
      }
    } catch (error) {
      console.error('Failed to load documents:', error)
    }
  }

  const loadDocument = (doc: any) => {
    if (doc.filename) {
      // Add timestamp to prevent caching issues
      const timestamp = Date.now()
      const pdfUrl = `/api/files/${doc.filename}?t=${timestamp}&cache=false`
      console.log('Loading document:', doc.filename, 'URL:', pdfUrl)
      setCurrentPdf(pdfUrl)
      
      // Clear any previous document info to force refresh
      setSelectedText("")
      setSearchResults([])
      setInsights({})
    }
  }

  const handleTextSelection = async (text: string) => {
    console.log('📝 Text selected for lens analysis:', text.substring(0, 100) + '...')
    
    if (text.trim().length < 5) {
      console.log('⚠️ Text too short for meaningful analysis (minimum 5 characters)')
      return
    }

    // Immediately show user feedback
    setSelectedText(text)
    setIsSearching(true)
    setSearchResults([])
    setInsights({})
    
    // Add visual feedback to the user
    console.log('🚀 Starting automatic analysis...')

    // Ensure text selection is enabled in PDF viewer
    if (pdfViewerRef.current) {
      pdfViewerRef.current.enableSelection(true)
    }

    try {
      console.log('🔍 Performing semantic search for lens selection...')
      // Step 1: Semantic search for related sections
      const searchResponse = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_text: text,
          max_results: 15, // Increased for better categorization
          context: "lens_selection", // Updated context
          include_metadata: true, // Request additional metadata for better categorization
          auto_trigger: true // Flag for auto-triggered analysis
        })
      })

      if (searchResponse.ok) {
        const results = await searchResponse.json()
        console.log('✅ Search results received:', results.length, 'matches')
        setSearchResults(results)
        
        // Switch to related tab to show results immediately
        setActiveTab("related")
        
        // Auto-generate insights immediately if we have results
        if (results.length > 0) {
          console.log('🧠 Auto-generating insights for lens selection...')
          setTimeout(() => {
            generateInsights()
          }, 500) // Reduced delay for faster response
        } else {
          console.log('💡 No matches found, providing guidance')
          setInsights({
            summary: `No direct matches found for "${text.substring(0, 50)}..."`,
            key_points: [
              "Try selecting more specific terms",
              "Look for technical terminology or key concepts", 
              "Select complete phrases rather than partial words"
            ],
            recommendations: ["Select different text areas to find related content"]
          })
        }
      } else {
        console.error('❌ Search failed:', searchResponse.status, searchResponse.statusText)
        const errorText = await searchResponse.text()
        console.error('Search error details:', errorText)
        
        // Provide helpful error feedback
        setInsights({
          summary: "Analysis temporarily unavailable",
          key_points: ["Search service encountered an issue"],
          recommendations: ["Try selecting text again", "Check if the document is fully loaded"]
        })
      }
    } catch (error) {
      console.error('❌ Analysis error:', error)
      setInsights({
        summary: "Failed to analyze selected text",
        key_points: [`Error: ${error instanceof Error ? error.message : 'Unknown error'}`],
        recommendations: ["Try selecting different text", "Refresh the page if issues persist"]
      })
    } finally {
      setIsSearching(false)
    }
  }

  const generateInsights = async () => {
    if (!selectedText || searchResults.length === 0) {
      console.log('No text selected or no search results available')
      return
    }

    setIsGeneratingInsights(true)

    try {
      console.log('Generating AI insights...')
      const response = await fetch('/api/insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_text: selectedText,
          related_sections: searchResults,
          insight_types: [
            'related',
            'overlapping', 
            'contradicting',
            'examples',
            'extensions',
            'problems',
            'applications',
            'methodology'
          ],
          analysis_depth: 'comprehensive'
        })
      })

      if (response.ok) {
        const result = await response.json()
        console.log('Insights generated successfully')
        setInsights(result.insights)
        setActiveTab("insights")
      } else {
        console.error('Insights generation failed:', response.status)
      }
    } catch (error) {
      console.error('Insights generation error:', error)
    } finally {
      setIsGeneratingInsights(false)
    }
  }

  const generatePodcast = async () => {
    if (!selectedText || searchResults.length === 0) {
      console.log('No text selected or no search results available')
      return
    }

    setIsGeneratingAudio(true)

    try {
      console.log('Generating podcast audio...')
      const response = await fetch('/api/audio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_text: selectedText,
          related_sections: searchResults,
          insights: insights,
          script_style: 'engaging_podcast',
          audio_format: {
            style: 'conversational',
            speakers: 2,
            duration_target: '3-5_minutes'
          }
        })
      })

      if (response.ok) {
        const result = await response.json()
        console.log('Podcast generated successfully')
        const files = result.audio_files || []
        setAudioSegments(files)

        // Auto-load the generated single podcast file (if present), do not auto-play
        if (files.length > 0 && audioRef.current) {
          // Add cache-busting query to avoid stale cached responses
          const url = `/api/audio/${files[0].filename}?t=${Date.now()}`
          audioRef.current.src = url
          // Force the browser to fetch metadata for the new source
          try { audioRef.current.load() } catch {}
          setCurrentAudioIndex(0)
          setIsPlaying(false)
          // Log when metadata is available (duration should be > 0)
          audioRef.current.onloadedmetadata = () => {
            console.log('Audio metadata loaded, duration:', audioRef.current?.duration)
          }
          audioRef.current.onerror = (e) => {
            console.error('Audio load error:', e)
          }
        }
      } else {
        console.error('Podcast generation failed:', response.status)
      }
    } catch (error) {
      console.error('Podcast generation error:', error)
    } finally {
      setIsGeneratingAudio(false)
    }
  }

  // Helper function to format snippet to show 2-3 lines
  const formatSnippet = (snippet: string, maxLines: number = 3) => {
    const sentences = snippet.split(/[.!?]+/).filter(s => s.trim().length > 0)
    if (sentences.length <= maxLines) {
      return snippet
    }
    // Take first 2-3 sentences and add ellipsis if truncated
    return sentences.slice(0, maxLines).join('. ').trim() + (sentences.length > maxLines ? '...' : '')
  }

  const handleJumpToHighlight = async (result: SearchResult, openInNewTab: boolean = false) => {
    console.log('Navigating to:', result.page_number, result.highlight_text, 'from document:', result.document_name)
    
    if (openInNewTab) {
      // Open in new tab
      const pdfUrl = `/api/files/${result.document_name}#page=${result.page_number}`
      window.open(pdfUrl, '_blank', 'noopener,noreferrer')
      console.log(`Opened "${result.section_title}" on page ${result.page_number} in new tab`)
      return
    }
    
    // Check if the result is from the current document
    const currentDocName = currentPdf ? documents.find(d => `/api/files/${d.filename}` === currentPdf.split('?')[0])?.filename : null
    
    if (currentDocName === result.document_name) {
      // Navigate within current PDF viewer
      navigateToPageAndHighlight(result)
    } else {
      // Load the document first, then navigate
      const targetDoc = documents.find(d => d.filename === result.document_name)
      if (targetDoc) {
        loadDocument(targetDoc)
        // Wait a moment for the PDF to load, then navigate
        setTimeout(() => {
          navigateToPageAndHighlight(result)
        }, 1000)
      } else {
        console.warn('Document not found in library:', result.document_name)
      }
    }
  }

  const navigateToPageAndHighlight = (result: SearchResult) => {
    if (pdfViewerRef.current) {
      console.log(`Navigating to page ${result.page_number} with text: "${result.highlight_text}"`)
      
      // Clear any previous selections first
      pdfViewerRef.current.clearSelection(result.page_number)
      
      // Navigate to the page with slight offset to center the content
      pdfViewerRef.current.jumpToPage(result.page_number, 50, 100)
      
      // Wait a brief moment for page navigation, then highlight
      setTimeout(() => {
        if (pdfViewerRef.current && result.highlight_text) {
          pdfViewerRef.current.highlightText(result.highlight_text, result.page_number)
        }
      }, 300)
      
      // Show a notification about the navigation
      console.log(`Navigated to "${result.section_title}" on page ${result.page_number}`)
    } else {
      console.warn('PDF viewer ref not available for navigation')
    }
  }

  const playAudioSegment = (index: number) => {
    if (audioRef.current && audioSegments[index]) {
      const audioUrl = `/api/audio/${audioSegments[index].filename}`
      audioRef.current.src = audioUrl
      audioRef.current.play()
      setCurrentAudioIndex(index)
      setIsPlaying(true)
    }
  }

  const togglePlayPause = async () => {
    const audio = audioRef.current
    if (!audio) return
    try {
      if (isPlaying) {
        audio.pause()
        setIsPlaying(false)
      } else {
        // Ensure metadata is loaded before playing to avoid 0.00 duration issues
        if (audio.readyState < 2) {
          await new Promise<void>((resolve, reject) => {
            const onLoaded = () => { audio.removeEventListener('loadeddata', onLoaded); resolve() }
            const onError = () => { audio.removeEventListener('error', onError); reject(new Error('Audio load error')) }
            audio.addEventListener('loadeddata', onLoaded, { once: true })
            audio.addEventListener('error', onError, { once: true })
            // Trigger load explicitly in case browser deferred it
            try { audio.load() } catch {}
          })
        }
        await audio.play()
        setIsPlaying(true)
      }
    } catch (err) {
      console.warn('Play/pause error:', err)
      setIsPlaying(false)
    }
  }

  // Categorize search results for different tabs
  // Use top-K strategy instead of absolute thresholds: MiniLM scores often sit in the 0.15-0.35 range,
  // so a hard threshold like 0.7 hides useful matches. Show the top results by similarity.
  const sortedResults = [...searchResults].sort((a, b) => b.similarity_score - a.similarity_score)
  const relatedResults = sortedResults.slice(0, Math.min(8, sortedResults.length))
  const overlappingResults = sortedResults.slice(8, Math.min(16, sortedResults.length))
  const contradictingResults = searchResults.filter(r => r.snippet.toLowerCase().includes('however') || r.snippet.toLowerCase().includes('but') || r.snippet.toLowerCase().includes('contrary'))
  const exampleResults = searchResults.filter(r => r.snippet.toLowerCase().includes('example') || r.snippet.toLowerCase().includes('case study') || r.snippet.toLowerCase().includes('instance'))

  return (
    <div className="min-h-screen flex flex-col lg:flex-row bg-gradient-to-br from-slate-50 to-white">
      {/* Left Panel - Adobe PDF Viewer */}
      <div className="flex-1 flex flex-col border-r border-slate-200 min-h-[50vh] lg:min-h-screen">
        {/* Header */}
        <div className="border-b border-slate-200 p-4 bg-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/">
                <Button variant="ghost" size="sm" className="justify-start p-0 h-auto text-slate-600 hover:text-slate-900">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Home
                </Button>
              </Link>
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-slate-600" />
                <span className="text-lg font-medium text-slate-900">
                  {currentPdf ? documents.find(d => `/api/files/${d.filename}` === currentPdf.split('?')[0])?.original_name || 'Document' : 'PDF Viewer'}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-sm text-slate-600">
                  {documents.filter(d => d.processing_status === 'completed').length} documents ready
                </span>
              </div>
              
              {/* Auto-Analysis Indicator */}
              <div className="flex items-center gap-2 ml-4 px-3 py-1 bg-green-50 border border-green-200 rounded-lg">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-green-700 font-medium">Auto-Analysis Active</span>
                <span className="text-xs text-green-600">Highlight any text to analyze</span>
              </div>
              
              {/* Search Status Indicator */}
              {isSearching && (
                <div className="flex items-center gap-2 ml-2 px-3 py-1 bg-orange-50 border border-orange-200 rounded-lg">
                  <div className="w-2 h-2 bg-orange-500 rounded-full animate-spin"></div>
                  <span className="text-sm text-orange-700 font-medium">Analyzing...</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* PDF Viewer */}
        <div className="flex-1">
          {currentPdf ? (
            <PDFViewer 
              ref={pdfViewerRef}
              onTextSelection={handleTextSelection} 
              selectedText={selectedText}
              pdfUrl={currentPdf}
              onDocumentLoad={(docInfo) => {
                console.log('Document loaded:', docInfo)
              }}
            />
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <FileText className="w-16 h-16 text-slate-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-900 mb-2">No PDF Loaded</h3>
                <p className="text-slate-600 mb-4">Select a document from the right panel</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right Panel - Analysis Tabs */}
      <div className="w-full lg:w-96 lg:max-w-md lg:min-w-[320px] flex flex-col bg-white border-l border-slate-200">
        {/* Top Controls */}
        <div className="border-b border-slate-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-serif text-lg font-semibold text-slate-900">Document Analysis</h2>
            <Badge variant={selectedText ? "default" : "secondary"} className="text-xs">
              {selectedText ? "Text Selected" : "No Selection"}
            </Badge>
          </div>
          
          {selectedText && (
            <div className="space-y-3">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-4 h-4 text-blue-600" />
                  <span className="text-sm font-medium text-blue-900">Selected Text</span>
                  <Badge variant="secondary" className="text-xs">
                    {selectedText.length} chars
                  </Badge>
                </div>
                <p className="text-sm text-blue-800 italic">
                  "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"
                </p>
              </div>
              
              <div className="flex gap-2">
                <Button
                  onClick={generateInsights}
                  disabled={isGeneratingInsights || searchResults.length === 0}
                  size="sm"
                  className="flex-1"
                >
                  {isGeneratingInsights ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4 mr-2" />
                      Generate Insights
                    </>
                  )}
                </Button>
                
                <Button
                  onClick={generatePodcast}
                  disabled={isGeneratingAudio || searchResults.length === 0}
                  size="sm"
                  variant="outline"
                  className="flex-1"
                >
                  {isGeneratingAudio ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    <>
                      <Volume2 className="w-4 h-4 mr-2" />
                      Play Podcast
                    </>
                  )}
                </Button>
              </div>

              {/* Audio Player */}
              {audioSegments.length > 0 && (
                <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Volume1 className="w-4 h-4 text-orange-600" />
                    <span className="text-sm font-medium text-orange-900">Podcast Ready</span>
                  </div>
                  <AudioPlayer audioUrl={`/api/audio/${audioSegments[0].filename}?t=${Date.now()}`} />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Analysis Tabs */}
        <div className="flex-1 overflow-hidden">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
            <TabsList className="grid w-full grid-cols-5 mx-4 mt-4">
              <TabsTrigger value="related" className="text-xs">Related</TabsTrigger>
              <TabsTrigger value="overlapping" className="text-xs">Overlap</TabsTrigger>
              <TabsTrigger value="contradicting" className="text-xs">Contrast</TabsTrigger>
              <TabsTrigger value="examples" className="text-xs">Examples</TabsTrigger>
              <TabsTrigger value="insights" className="text-xs">Insights</TabsTrigger>
            </TabsList>

            <div className="flex-1 overflow-hidden">
              <ScrollArea className="h-full px-4 py-4">
                {/* Related Tab */}
                <TabsContent value="related" className="space-y-3 mt-0">
                  {isSearching ? (
                    <div className="flex items-center justify-center p-8">
                      <div className="flex flex-col items-center gap-3">
                        <Loader2 className="animate-spin h-8 w-8 text-blue-600" />
                        <p className="text-sm text-slate-600">Searching documents...</p>
                      </div>
                    </div>
                  ) : relatedResults.length > 0 ? (
                    <>
                      <div className="flex items-center gap-2 mb-3">
                        <Search className="w-4 h-4 text-green-600" />
                        <span className="font-medium text-slate-900">Related Sections</span>
                        <Badge variant="secondary" className="text-xs">
                          {relatedResults.length} found
                        </Badge>
                      </div>
                      {relatedResults.map((result, index) => (
                        <Card 
                          key={result.id} 
                          className="cursor-pointer hover:shadow-lg transition-all duration-200 hover:scale-[1.02] border-l-4 border-l-green-500 hover:border-l-green-600"
                          onClick={() => handleJumpToHighlight(result)}
                          onAuxClick={(e) => {
                            // Middle click opens in new tab
                            if (e.button === 1) {
                              e.preventDefault()
                              handleJumpToHighlight(result, true)
                            }
                          }}
                          onKeyDown={(e) => {
                            // Ctrl+Click or Cmd+Click opens in new tab
                            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                              e.preventDefault()
                              handleJumpToHighlight(result, true)
                            } else if (e.key === 'Enter') {
                              handleJumpToHighlight(result)
                            }
                          }}
                          title={`Click to view in current PDF viewer (Page ${result.page_number}). Ctrl+Click or use 'New Tab' button to open in new tab.`}
                        >
                          <CardContent className="p-3">
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <Badge variant="outline" className="text-xs truncate max-w-[120px]">
                                  {result.document_name}
                                </Badge>
                                <Badge variant="secondary" className="text-xs">
                                  {(result.similarity_score * 100).toFixed(0)}%
                                </Badge>
                              </div>
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-1 text-xs text-green-600">
                                  <Eye className="w-3 h-3" />
                                  <span>View in PDF</span>
                                </div>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-6 px-2 text-xs text-slate-500 hover:text-blue-600"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleJumpToHighlight(result, true)
                                  }}
                                >
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  New Tab
                                </Button>
                              </div>
                              <h4 className="font-medium text-slate-900 text-sm leading-tight line-clamp-2">{result.section_title}</h4>
                              <div className="text-xs text-slate-600 space-y-1">
                                <p className="line-clamp-4 leading-relaxed">{formatSnippet(result.snippet, 3)}</p>
                              </div>
                              <div className="flex items-center gap-2 text-xs text-slate-500">
                                <FileText className="w-3 h-3" />
                                Page {result.page_number}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </>
                  ) : selectedText ? (
                    <div className="text-center p-8">
                      <Search className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                      <p className="text-slate-600 mb-2">No related sections found</p>
                      <p className="text-xs text-slate-500">Try selecting different text</p>
                    </div>
                  ) : (
                    <div className="text-center p-8">
                      <Target className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                      <p className="text-slate-600 mb-2">Select text in the PDF to find related sections</p>
                      <p className="text-xs text-slate-500">AI will search across your document library</p>
                    </div>
                  )}
                </TabsContent>

                {/* Overlapping Tab */}
                <TabsContent value="overlapping" className="space-y-3 mt-0">
                  {overlappingResults.length > 0 ? (
                    <>
                      <div className="flex items-center gap-2 mb-3">
                        <Sparkles className="w-4 h-4 text-blue-600" />
                        <span className="font-medium text-slate-900">Overlapping Information</span>
                        <Badge variant="secondary" className="text-xs">
                          {overlappingResults.length} found
                        </Badge>
                      </div>
                      {overlappingResults.map((result, index) => (
                        <Card 
                          key={result.id} 
                          className="cursor-pointer hover:shadow-md transition-all duration-200 hover:scale-[1.02] border-l-4 border-l-blue-500"
                          onClick={() => handleJumpToHighlight(result)}
                        >
                          <CardContent className="p-3">
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <Badge variant="outline" className="text-xs truncate max-w-[120px]">
                                  {result.document_name}
                                </Badge>
                                <Badge className="text-xs bg-blue-100 text-blue-700">
                                  {(result.similarity_score * 100).toFixed(0)}%
                                </Badge>
                              </div>
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-1 text-xs text-blue-600">
                                  <Eye className="w-3 h-3" />
                                  <span>View in PDF</span>
                                </div>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-6 px-2 text-xs text-slate-500 hover:text-blue-600"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleJumpToHighlight(result, true)
                                  }}
                                >
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  New Tab
                                </Button>
                              </div>
                              <h4 className="font-medium text-slate-900 text-sm leading-tight line-clamp-2">{result.section_title}</h4>
                              <div className="text-xs text-slate-600 space-y-1">
                                <p className="line-clamp-4 leading-relaxed">{formatSnippet(result.snippet, 3)}</p>
                              </div>
                              <div className="flex items-center gap-2 text-xs text-slate-500">
                                <FileText className="w-3 h-3" />
                                Page {result.page_number}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </>
                  ) : (
                    <div className="text-center p-8">
                      <Sparkles className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                      <p className="text-slate-600 mb-2">No overlapping information found</p>
                      <p className="text-xs text-slate-500">Select text to find overlapping concepts</p>
                    </div>
                  )}
                </TabsContent>

                {/* Contradicting Tab */}
                <TabsContent value="contradicting" className="space-y-3 mt-0">
                  {contradictingResults.length > 0 ? (
                    <>
                      <div className="flex items-center gap-2 mb-3">
                        <AlertCircle className="w-4 h-4 text-red-600" />
                        <span className="font-medium text-slate-900">Contradictory Findings</span>
                        <Badge variant="secondary" className="text-xs">
                          {contradictingResults.length} found
                        </Badge>
                      </div>
                      {contradictingResults.map((result, index) => (
                        <Card 
                          key={result.id} 
                          className="cursor-pointer hover:shadow-md transition-all duration-200 hover:scale-[1.02] border-l-4 border-l-red-500"
                          onClick={() => handleJumpToHighlight(result)}
                        >
                          <CardContent className="p-3">
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <Badge variant="outline" className="text-xs truncate max-w-[120px]">
                                  {result.document_name}
                                </Badge>
                                <Badge className="text-xs bg-red-100 text-red-700">
                                  Contradictory
                                </Badge>
                              </div>
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-1 text-xs text-red-600">
                                  <Eye className="w-3 h-3" />
                                  <span>View in PDF</span>
                                </div>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-6 px-2 text-xs text-slate-500 hover:text-blue-600"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleJumpToHighlight(result, true)
                                  }}
                                >
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  New Tab
                                </Button>
                              </div>
                              <h4 className="font-medium text-slate-900 text-sm leading-tight line-clamp-2">{result.section_title}</h4>
                              <div className="text-xs text-slate-600 space-y-1">
                                <p className="line-clamp-4 leading-relaxed">{formatSnippet(result.snippet, 3)}</p>
                              </div>
                              <div className="flex items-center gap-2 text-xs text-slate-500">
                                <FileText className="w-3 h-3" />
                                Page {result.page_number}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </>
                  ) : (
                    <div className="text-center p-8">
                      <AlertCircle className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                      <p className="text-slate-600 mb-2">No contradictory findings detected</p>
                      <p className="text-xs text-slate-500">Select text to find opposing viewpoints</p>
                    </div>
                  )}
                </TabsContent>

                {/* Examples Tab */}
                <TabsContent value="examples" className="space-y-3 mt-0">
                  {exampleResults.length > 0 ? (
                    <>
                      <div className="flex items-center gap-2 mb-3">
                        <Lightbulb className="w-4 h-4 text-purple-600" />
                        <span className="font-medium text-slate-900">Examples & Case Studies</span>
                        <Badge variant="secondary" className="text-xs">
                          {exampleResults.length} found
                        </Badge>
                      </div>
                      {exampleResults.map((result, index) => (
                        <Card 
                          key={result.id} 
                          className="cursor-pointer hover:shadow-md transition-all duration-200 hover:scale-[1.02] border-l-4 border-l-purple-500"
                          onClick={() => handleJumpToHighlight(result)}
                        >
                          <CardContent className="p-3">
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <Badge variant="outline" className="text-xs truncate max-w-[120px]">
                                  {result.document_name}
                                </Badge>
                                <Badge className="text-xs bg-purple-100 text-purple-700">
                                  Example
                                </Badge>
                              </div>
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-1 text-xs text-purple-600">
                                  <Eye className="w-3 h-3" />
                                  <span>View in PDF</span>
                                </div>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-6 px-2 text-xs text-slate-500 hover:text-blue-600"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleJumpToHighlight(result, true)
                                  }}
                                >
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  New Tab
                                </Button>
                              </div>
                              <h4 className="font-medium text-slate-900 text-sm leading-tight line-clamp-2">{result.section_title}</h4>
                              <div className="text-xs text-slate-600 space-y-1">
                                <p className="line-clamp-4 leading-relaxed">{formatSnippet(result.snippet, 3)}</p>
                              </div>
                              <div className="flex items-center gap-2 text-xs text-slate-500">
                                <FileText className="w-3 h-3" />
                                Page {result.page_number}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </>
                  ) : (
                    <div className="text-center p-8">
                      <Lightbulb className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                      <p className="text-slate-600 mb-2">No examples found</p>
                      <p className="text-xs text-slate-500">Select text to find related examples</p>
                    </div>
                  )}
                </TabsContent>

                {/* Insights Tab */}
                <TabsContent value="insights" className="space-y-3 mt-0">
                  {Object.keys(insights).length > 0 ? (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 mb-3">
                        <Brain className="w-4 h-4 text-purple-600" />
                        <span className="font-medium text-slate-900">AI Analysis & Insights</span>
                        <Badge variant="outline" className="text-xs">
                          Generated by Gemini
                        </Badge>
                      </div>

                      {insights.related && (
                        <Card className="bg-green-50 border-green-200">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-sm text-green-800 flex items-center gap-2">
                              <Lightbulb className="w-4 h-4" />
                              Related Methods & Concepts
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="pt-0">
                            <p className="text-sm text-green-700">{insights.related}</p>
                          </CardContent>
                        </Card>
                      )}

                      {insights.overlapping && (
                        <Card className="bg-blue-50 border-blue-200">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-sm text-blue-800 flex items-center gap-2">
                              <Sparkles className="w-4 h-4" />
                              Overlapping Information
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="pt-0">
                            <p className="text-sm text-blue-700">{insights.overlapping}</p>
                          </CardContent>
                        </Card>
                      )}

                      {insights.contradicting && insights.contradicting !== "No contradictions found." && (
                        <Card className="bg-red-50 border-red-200">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-sm text-red-800 flex items-center gap-2">
                              <AlertCircle className="w-4 h-4" />
                              Contradictory Findings
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="pt-0">
                            <p className="text-sm text-red-700">{insights.contradicting}</p>
                          </CardContent>
                        </Card>
                      )}

                      {insights.examples && (
                        <Card className="bg-purple-50 border-purple-200">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-sm text-purple-800 flex items-center gap-2">
                              <FileText className="w-4 h-4" />
                              Examples & Case Studies
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="pt-0">
                            <p className="text-sm text-purple-700">{insights.examples}</p>
                          </CardContent>
                        </Card>
                      )}

                      {insights.extensions && (
                        <Card className="bg-indigo-50 border-indigo-200">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-sm text-indigo-800 flex items-center gap-2">
                              <Target className="w-4 h-4" />
                              Extensions & Applications
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="pt-0">
                            <p className="text-sm text-indigo-700">{insights.extensions}</p>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  ) : isGeneratingInsights ? (
                    <div className="flex items-center justify-center p-8">
                      <div className="flex flex-col items-center gap-3">
                        <Loader2 className="animate-spin h-8 w-8 text-purple-600" />
                        <p className="text-sm text-slate-600">Generating AI insights...</p>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center p-8">
                      <Brain className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                      <p className="text-slate-600 mb-2">No insights generated yet</p>
                      <p className="text-xs text-slate-500 mb-4">Select text and click "Generate Insights" above</p>
                      <Button
                        onClick={generateInsights}
                        disabled={!selectedText || searchResults.length === 0}
                        size="sm"
                      >
                        <Brain className="w-4 h-4 mr-2" />
                        Generate Insights
                      </Button>
                    </div>
                  )}
                </TabsContent>
              </ScrollArea>
            </div>
          </Tabs>
        </div>

        {/* Document Library */}
        <div className="border-t border-slate-200 p-4 bg-slate-50">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-slate-900">Document Library</span>
            <Badge variant="outline" className="text-xs">
              {documents.length} documents
            </Badge>
          </div>
          
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            multiple
            onChange={async (event) => {
              const files = Array.from(event.target.files || [])
              if (files.length === 0) return

              setIsUploading(true)
              const formData = new FormData()
              files.forEach(file => formData.append('files', file))

              try {
                const response = await fetch('/api/upload', {
                  method: 'POST',
                  body: formData
                })
                if (response.ok) {
                  await loadDocuments()
                }
              } catch (error) {
                console.error('Upload failed:', error)
              } finally {
                setIsUploading(false)
              }
            }}
            className="hidden"
          />

          <div className="space-y-2 max-h-40 overflow-y-auto">
            {documents.length === 0 ? (
              <Button
                onClick={() => fileInputRef.current?.click()}
                variant="outline"
                className="w-full text-xs h-8"
                disabled={isUploading}
              >
                <Upload className="w-3 h-3 mr-2" />
                {isUploading ? 'Uploading...' : 'Upload PDFs'}
              </Button>
            ) : (
              <>
                {documents.map((doc, index) => (
                  <Card 
                    key={doc.id} 
                    className={`cursor-pointer transition-all duration-200 ${
                      currentPdf && currentPdf.includes(doc.filename)
                        ? 'bg-blue-50 border-blue-200 shadow-sm' 
                        : 'bg-white hover:bg-slate-50 border-slate-200'
                    }`}
                    onClick={() => loadDocument(doc)}
                  >
                    <CardContent className="p-3">
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium text-slate-900 truncate">
                            {doc.title || doc.filename}
                          </p>
                          <p className="text-xs text-slate-600">
                            {doc.total_sections} sections
                          </p>
                        </div>
                        <Badge 
                          variant={doc.processing_status === 'completed' ? 'default' : 'secondary'}
                          className="text-xs ml-2"
                        >
                          {doc.processing_status === 'completed' ? '✓' : '...'}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                ))}
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  variant="outline"
                  className="w-full text-xs h-8"
                  disabled={isUploading}
                >
                  <Upload className="w-3 h-3 mr-2" />
                  Add More PDFs
                </Button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}