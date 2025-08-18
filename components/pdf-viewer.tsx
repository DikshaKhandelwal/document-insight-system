"use client"

import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Upload, FileText, Search, ZoomIn, ZoomOut, RotateCw, AlertCircle, CheckCircle } from "lucide-react"

interface PDFViewerProps {
  onTextSelection: (text: string) => void
  selectedText: string
  pdfUrl?: string
  onDocumentLoad?: (documentInfo: any) => void
}

interface PDFViewerRef {
  jumpToPage: (page: number) => void
  highlightText: (text: string, page: number) => void
  getSelectedText: () => string
}

// Adobe PDF Embed API types
declare global {
  interface Window {
    AdobeDC: any
  }
}

interface DocumentInfo {
  fileName: string
  numPages: number
  title?: string
}

// Helper: normalize various Adobe selected-content response shapes into text
const extractSelectedTextFromResult = (res: any): string => {
  try {
    if (!res) return ''
    if (typeof res === 'string') return res.trim()
    // Common shape: { data: [ { content: '...' }, ... ] }
    if (res.data) {
      if (Array.isArray(res.data)) {
        return res.data.map((it: any) => (it.content || it.text || it.value || '')).join(' ').trim()
      }
      if (typeof res.data === 'string') return res.data.trim()
      if (res.data.content) return (res.data.content || res.data.text || '').trim()
    }
    // fallback to content/text/value
    if (res.content) return (res.content || '').trim()
    if (res.text) return (res.text || '').trim()
    return ''
  } catch (e) {
    console.error('extractSelectedTextFromResult error', e)
    return ''
  }
}

// Helper: extract text from event.data when annotation or other events supply ranges
const extractSelectedTextFromEventData = (data: any): string => {
  try {
    if (!data) return ''
    // Some events include a "selectedText" field
    if (data.selectedText) return (data.selectedText || '').trim()
    // Some annotation objects may contain a "contents" or "text" property
    if (data.contents) return (data.contents || data.text || '').trim()
    if (data.text) return (data.text || '').trim()
    // Try nested structures
    if (Array.isArray(data)) return data.map(d => (d.text || d.content || '')).join(' ').trim()
    return ''
  } catch (e) {
    console.error('extractSelectedTextFromEventData error', e)
    return ''
  }
}

const PDFViewer = forwardRef<PDFViewerRef, PDFViewerProps>(({ onTextSelection, selectedText, pdfUrl, onDocumentLoad }, ref) => {
  const viewerContainerRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [adobeViewer, setAdobeViewer] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [loadedPdfUrl, setLoadedPdfUrl] = useState<string | null>(null)
  const [documentInfo, setDocumentInfo] = useState<DocumentInfo | null>(null)
  const [useAdobeEmbed, setUseAdobeEmbed] = useState(true)
  const [apiStatus, setApiStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [manualText, setManualText] = useState('')
  const [lastSelectedText, setLastSelectedText] = useState('')

  // Adobe PDF Embed API Client ID - Get from Adobe Developer Console
  const ADOBE_CLIENT_ID = process.env.NEXT_PUBLIC_ADOBE_PDF_EMBED_CLIENT_ID || ""

  // Expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    jumpToPage: (page: number, x?: number, y?: number) => {
      // Return the promise so callers can await navigation
      return goToLocation(page, undefined, x, y)
    },
    gotoLocation: (page: number, x?: number, y?: number) => {
      return goToLocation(page, undefined, x, y)
    },
    highlightText: (text: string, page: number) => {
      // Navigate to page and clear previous selections
      console.log('Highlighting text:', text, 'on page:', page)
      clearPageSelection(page)
      // Try to navigate to the page and then search/highlight
      goToLocation(page, text).catch((e) => console.warn('gotoLocation failed during highlight:', e))
      // Enable text selection to make sure user can interact
      enableTextSelection(true)
    },
    getSelectedText: () => {
      return selectedText
    },
    clearSelection: (page: number) => {
      clearPageSelection(page)
    },
    enableSelection: (enabled: boolean) => {
      enableTextSelection(enabled)
    }
  }), [selectedText, adobeViewer, loadedPdfUrl])

  useEffect(() => {
    console.log('PDF Viewer Environment Check:')
    console.log('- Adobe Client ID:', ADOBE_CLIENT_ID ? 'Set' : 'Not set')
    console.log('- Use Adobe Embed:', useAdobeEmbed)
    loadAdobePDFEmbed()
    
    // Remove global text selection listeners - they interfere with Adobe PDF selection
    // Adobe PDF Embed API will handle text selection internally via callbacks
  }, [])

  useEffect(() => {
    if (pdfUrl && pdfUrl !== loadedPdfUrl) {
      console.log('PDF URL changed, force reloading:', pdfUrl)
      setLoadedPdfUrl(null) // Clear current PDF first
      setDocumentInfo(null) // Clear document info
      
      // Small delay to ensure state is cleared
      setTimeout(() => {
        setLoadedPdfUrl(pdfUrl)
        
        // Try Adobe if available, otherwise use iframe
        if (adobeViewer && apiStatus === 'ready') {
          loadPDFWithAdobe(pdfUrl)
        }
      }, 100)
    }
  }, [pdfUrl, adobeViewer, apiStatus, loadedPdfUrl])

  const loadAdobePDFEmbed = async () => {
    try {
      setApiStatus('loading')
      
      // Check if Adobe PDF Embed API is already loaded
      if (window.AdobeDC) {
        initializeAdobeViewer()
        return
      }

      // Check if we have a valid client ID before loading script
      if (!ADOBE_CLIENT_ID || ADOBE_CLIENT_ID === "" || ADOBE_CLIENT_ID === "your_adobe_client_id_here") {
        console.warn('Adobe PDF Embed API Client ID not configured. Using demo mode.')
        setApiStatus('error')
        setUseAdobeEmbed(false)
        return
      }

      console.log('Loading Adobe PDF Embed API script...')
      
      // Load Adobe PDF Embed API script
      const script = document.createElement('script')
      script.src = 'https://documentservices.adobe.com/view-sdk/viewer.js'
      
      // Add timeout for script loading
      const timeout = setTimeout(() => {
        console.error('Adobe PDF Embed API script load timeout')
        setApiStatus('error')
        setUseAdobeEmbed(false)
      }, 10000) // 10 second timeout
      
      script.onload = () => {
        clearTimeout(timeout)
        console.log('Adobe PDF Embed API script loaded successfully')
        // Wait a moment for the API to be fully available
        setTimeout(() => {
          if (window.AdobeDC) {
            initializeAdobeViewer()
          } else {
            console.error('Adobe DC not available after script load')
            setApiStatus('error')
            setUseAdobeEmbed(false)
          }
        }, 500)
      }
      
      script.onerror = () => {
        clearTimeout(timeout)
        console.error('Failed to load Adobe PDF Embed API script')
        setApiStatus('error')
        setUseAdobeEmbed(false)
      }
      
      document.head.appendChild(script)
    } catch (error) {
      console.error('Error loading Adobe PDF Embed API:', error)
      setApiStatus('error')
      setUseAdobeEmbed(false)
    }
  }

  const initializeAdobeViewer = () => {
    if (!window.AdobeDC) {
      setApiStatus('error')
      setUseAdobeEmbed(false)
      return
    }

    try {
      // Check if we have a valid client ID
      if (!ADOBE_CLIENT_ID || ADOBE_CLIENT_ID === "" || ADOBE_CLIENT_ID === "your_adobe_client_id_here") {
        console.warn('Adobe PDF Embed API Client ID not configured. Using demo mode.')
        setApiStatus('error')
        setUseAdobeEmbed(false)
        return
      }

      console.log('Initializing Adobe viewer for new PDF...')
      
      // Clear any existing viewer
      if (viewerContainerRef.current) {
        viewerContainerRef.current.innerHTML = '<div id="adobe-dc-view-container" style="height: 100%; width: 100%;"></div>'
      }
      
      // Initialize Adobe DC View
      const adobeDCView = new window.AdobeDC.View({
        clientId: ADOBE_CLIENT_ID,
        divId: "adobe-dc-view-container"
      })

      setAdobeViewer(adobeDCView)
      setApiStatus('ready')
      
      console.log('Adobe PDF Embed API initialized successfully')
    } catch (error) {
      console.error('Failed to initialize Adobe viewer:', error)
      setApiStatus('error')
      setUseAdobeEmbed(false)
    }
  }

  const loadPDFWithAdobe = async (url: string) => {
    if (!adobeViewer || !viewerContainerRef.current) {
      console.error('Adobe viewer not ready')
      return
    }

    setIsLoading(true)
    
    try {
      console.log('Loading PDF with Adobe:', url)
      
      // Force clear the viewer container first
      if (viewerContainerRef.current) {
        viewerContainerRef.current.innerHTML = '<div id="adobe-dc-view-container" style="height: 100%; width: 100%;"></div>'
      }
      
      // Re-initialize viewer for fresh load
      const adobeDCView = new window.AdobeDC.View({
        clientId: ADOBE_CLIENT_ID,
        divId: "adobe-dc-view-container"
      })
      
      setAdobeViewer(adobeDCView)
      
      // Configure viewer options with unique ID to force refresh
      const previewConfig = {
        // FULL_WINDOW exposes the annotation/selection toolset and APIs reliably
        embedMode: "FULL_WINDOW",
        showAnnotationTools: true,
        showLeftHandPanel: true,
        showDownloadPDF: true,
        showPrintPDF: true,
        showZoomControl: true,
        showPageControls: true,
        enableTextSelection: true,
        focusOnRendering: true,
        defaultViewMode: "FIT_PAGE"
      }

      // Create unique document ID to prevent caching
      const documentId = "document-" + Date.now() + "-" + Math.random().toString(36).substr(2, 9)

      // Load the PDF with cache-busting URL
      const cacheUrl = url.includes('?') ? `${url}&v=${Date.now()}` : `${url}?v=${Date.now()}`
      
      // Register event listener on adobeDCView BEFORE calling previewFile (per Adobe docs)
      try {
        adobeDCView.registerCallback(
          window.AdobeDC.View.Enum.CallbackType.EVENT_LISTENER,
          function(event: any) {
            console.log('🎯 Adobe event received:', event.type)
            
            if (event.type === "PREVIEW_SELECTION_END") {
              console.log('📋 PREVIEW_SELECTION_END detected - getting selected content')
              
              // Use the Adobe recommended pattern: previewFilePromise.then(adobeViewer => adobeViewer.getAPIs()...)
              previewFilePromise.then((adobeViewer: any) => {
                adobeViewer.getAPIs().then((apis: any) => {
                  console.log('📡 Got APIs, calling getSelectedContent()')
                  
                  apis.getSelectedContent()
                    .then((result: any) => {
                      console.log('📝 getSelectedContent result:', result)
                      console.log('📝 Raw result (stringified):', JSON.stringify(result, null, 2))
                      
                      // Extract text from the result object
                      let selectedText = ''
                      
                      if (result && result.data) {
                        if (typeof result.data === 'string') {
                          // Direct string in data field
                          selectedText = result.data.trim()
                          console.log('📄 Extracted from result.data string:', selectedText)
                        } else if (Array.isArray(result.data)) {
                          // Array format: { data: [{ content: "text" }] }
                          selectedText = result.data.map((item: any) => item.content || item.text || '').join(' ').trim()
                          console.log('📄 Extracted from result.data array:', selectedText)
                        }
                      } else if (result && result.text) {
                        selectedText = result.text.trim()
                        console.log('📄 Extracted from result.text:', selectedText)
                      } else if (result && typeof result === 'string') {
                        selectedText = result.trim()
                        console.log('📄 Extracted from string result:', selectedText)
                      }
                      
                      if (selectedText && selectedText.length > 0) {
                        console.log('✅ SELECTED TEXT:', selectedText)
                        console.log('✅ Text length:', selectedText.length)
                        console.log('==========================================')
                        console.log('FULL SELECTED TEXT:')
                        console.log(selectedText)
                        console.log('==========================================')
                        onTextSelection(selectedText)
                      } else {
                        console.log('❌ No text content found in selection result')
                        console.log('❌ Result keys:', Object.keys(result || {}))
                        console.log('❌ Result type:', typeof result)
                      }
                    })
                    .catch((error: any) => console.warn('❌ getSelectedContent failed:', error))
                }).catch((error: any) => console.warn('❌ getAPIs failed:', error))
              }).catch((error: any) => console.warn('❌ previewFilePromise failed:', error))
            }
            
            // Log other events for debugging
            else {
              console.log('📊 Other Adobe event:', event.type)
            }
          },
          { enableFilePreviewEvents: true }
        )
        console.log('✅ Adobe event listener registered on adobeDCView')
      } catch (error) {
        console.warn('❌ Could not register Adobe event listener on adobeDCView:', error)
      }
      
  const previewFilePromise = adobeDCView.previewFile({
        content: { location: { url: cacheUrl } },
        metaData: { 
          fileName: url.split('/').pop()?.split('?')[0] || "Document.pdf",
          id: documentId
        }
      }, previewConfig)

      // Register callbacks after preview is ready
  previewFilePromise.then(async (viewer: any) => {
        console.log('Adobe PDF preview ready, registering callbacks...', viewer)
        setLoadedPdfUrl(url)
        
        // Store the viewer instance for navigation methods
        setAdobeViewer(viewer)
        
        // Register event listener for selection and annotation events.
        // We listen for TEXT_SELECTION_CHANGED and PREVIEW_SELECTION_END (selection finalization)
        // and for ANNOTATION_ADDED so highlights created via the UI are captured.
        try {
          viewer.registerCallback(
            window.AdobeDC.View.Enum.CallbackType.EVENT_LISTENER,
            async function(event: any) {
              try {
                console.log('Adobe event received:', event.type, JSON.parse(JSON.stringify(event)))

                // Helper: attempt to extract from event.data using known fields
                const tryExtractFromEvent = (): string => {
                  try {
                    if (!event || !event.data) return ''
                    const d = event.data
                    if (d.selectedText) return (d.selectedText || '').trim()
                    if (d.selectionText) return (d.selectionText || '').trim()
                    if (d.text) return (d.text || '').trim()
                    if (d.contents) return (d.contents || '').trim()
                    // try nested structures
                    if (Array.isArray(d)) return d.map((it: any) => it.text || it.content || '').join(' ').trim()
                    return ''
                  } catch (e) {
                    return ''
                  }
                }

                // 1) Direct extraction from event.data (TEXT_SELECTION_CHANGED etc.)
                const direct = tryExtractFromEvent()
                if (direct && direct.length > 10) {
                  console.log('Extracted selection directly from event.data')
                  setLastSelectedText(direct.substring(0, 50) + '...')
                  onTextSelection(direct)
                  return
                }

                // 2) If selection finalized, try viewer.getSelectedContent (some API versions)
                if (event.type === 'PREVIEW_SELECTION_END' || event.type === 'TEXT_SELECTION_CHANGED') {
                  try {
                    // prefer direct method if present
                    if (typeof viewer.getSelectedContent === 'function') {
                      const res = await viewer.getSelectedContent()
                      const selected = extractSelectedTextFromResult(res)
                      if (selected && selected.length > 10) {
                        console.log('Extracted selection via viewer.getSelectedContent')
                        setLastSelectedText(selected.substring(0, 50) + '...')
                        onTextSelection(selected)
                        return
                      }
                    }

                    // fallback: some implementations expose getAPIs -> getSelectedContent
                    if (typeof viewer.getAPIs === 'function') {
                      try {
                        const apis = await viewer.getAPIs()
                        if (apis && typeof apis.getSelectedContent === 'function') {
                          const res2 = await apis.getSelectedContent()
                          const sel2 = extractSelectedTextFromResult(res2)
                          if (sel2 && sel2.length > 10) {
                            console.log('Extracted selection via viewer.getAPIs().getSelectedContent')
                            setLastSelectedText(sel2.substring(0, 50) + '...')
                            onTextSelection(sel2)
                            return
                          }
                        }
                      } catch (e) {
                        console.warn('viewer.getAPIs() fallback failed', e)
                      }
                    }
                  } catch (e) {
                    console.warn('Error while trying getSelectedContent fallback:', e)
                  }
                }

                // 3) Annotation events may include annotation payloads
                if (event.type && event.type.indexOf('ANNOTATION') >= 0) {
                  const maybeText = extractSelectedTextFromEventData(event.data)
                  if (maybeText && maybeText.length > 10) {
                    console.log('Extracted selection from annotation event')
                    setLastSelectedText(maybeText.substring(0, 50) + '...')
                    onTextSelection(maybeText)
                    return
                  }
                }

                // Nothing useful extracted — log for debugging. Include a truncated snapshot of event.data.
                try {
                  const snapshot = JSON.stringify(event.data || {})
                  const truncated = snapshot.length > 2000 ? snapshot.substring(0, 2000) + '... (truncated)' : snapshot
                  console.log('No selection text extracted from event; event.data keys:', Object.keys(event.data || {}), 'data snapshot:', truncated)
                } catch (e) {
                  console.log('No selection text extracted from event; could not stringify event.data')
                }
              } catch (e) {
                console.error('Error handling Adobe event:', e)
              }
            },
            {
              listenOn: window.AdobeDC.View.Enum.ListenerType.EVENT_LISTENER,
              enableFilePreviewEvents: true,
              events: [
                window.AdobeDC.View.Enum.Events.TEXT_SELECTION_CHANGED,
                'PREVIEW_SELECTION_END',
                'ANNOTATION_ADDED',
                'ANNOTATION_MODIFIED'
              ]
            }
          )
        } catch (error) {
          console.warn('Could not register Adobe event listener:', error)
        }

        // Register callback for document events
        try {
          viewer.registerCallback(
            window.AdobeDC.View.Enum.CallbackType.EVENT_LISTENER,
            function(event: any) {
              console.log('Document event received:', event.type, event)
              if (event.type === "DOCUMENT_LOADED") {
                const info: DocumentInfo = {
                  fileName: event.data?.fileName || "Document.pdf",
                  numPages: event.data?.numPages || 0,
                  title: event.data?.title
                }
                setDocumentInfo(info)
                if (onDocumentLoad) {
                  onDocumentLoad(info)
                }
              }
            },
            {
              listenOn: window.AdobeDC.View.Enum.ListenerType.EVENT_LISTENER,
              events: ["DOCUMENT_LOADED"]
            }
          )
        } catch (error) {
          console.warn('Could not register document load callback:', error)
        }

        // Attempt to enable text selection for this viewer instance right after preview loads.
        try {
          if (typeof (viewer as any).enableTextSelection === 'function') {
            try {
              await (viewer as any).enableTextSelection(true)
              console.log('Called viewer.enableTextSelection(true)')
            } catch (e) {
              console.warn('viewer.enableTextSelection failed', e)
            }
          } else if (typeof (viewer as any).getAPIs === 'function') {
            try {
              const apis = await (viewer as any).getAPIs()
              if (apis && typeof apis.enableTextSelection === 'function') {
                await apis.enableTextSelection(true)
                console.log('Called apis.enableTextSelection(true) via getAPIs')
              }
            } catch (e) {
              console.warn('getAPIs().enableTextSelection failed', e)
            }
          } else {
            console.log('No enableTextSelection API found on viewer instance')
          }
        } catch (e) {
          console.warn('Error while attempting to enable text selection on viewer instance:', e)
        }
      }).catch((error: any) => {
        console.error('Error setting up Adobe PDF preview:', error)
        // Fall back to iframe mode
        setUseAdobeEmbed(false)
        setLoadedPdfUrl(url)
      })

      setLoadedPdfUrl(url)
      console.log('PDF loaded successfully with Adobe')
      
    } catch (error) {
      console.error('Error loading PDF with Adobe:', error)
      // Fallback to demo mode if Adobe fails
      setUseAdobeEmbed(false)
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file || file.type !== 'application/pdf') {
      return
    }

    setIsUploading(true)
    setUploadProgress(0)

    // Create form data for upload
    const formData = new FormData()
    formData.append('files', file)

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 100)

      // Upload to backend
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      })

      clearInterval(progressInterval)
      setUploadProgress(100)

      if (response.ok) {
        const result = await response.json()
        console.log('File uploaded successfully:', result)
        
        // Create blob URL for immediate viewing
        const fileUrl = URL.createObjectURL(file)
        
        if (useAdobeEmbed && adobeViewer && apiStatus === 'ready') {
          await loadPDFWithAdobe(fileUrl)
        } else {
          setLoadedPdfUrl(fileUrl)
        }
      } else {
        throw new Error('Upload failed')
      }
    } catch (error) {
      console.error('Upload failed:', error)
    } finally {
      setIsUploading(false)
      setTimeout(() => setUploadProgress(0), 2000)
    }
  }

  const goToLocation = async (pageNumber: number, searchText?: string, x?: number, y?: number): Promise<void> => {
    if (!adobeViewer || !loadedPdfUrl) {
      console.warn('Adobe viewer not ready for navigation')
      return Promise.reject(new Error('Adobe viewer not ready'))
    }

    try {
      console.log(`Navigating to page ${pageNumber}`, { searchText, x, y })

      const xCoord = x || 0
      const yCoord = y || 0

      // Try direct method on viewer instance first
      if (typeof (adobeViewer as any).gotoLocation === 'function') {
        try {
          // Some SDK variants accept positional args, others accept an object
          const res = (adobeViewer as any).gotoLocation.length >= 3
            ? (adobeViewer as any).gotoLocation(pageNumber, xCoord, yCoord)
            : (adobeViewer as any).gotoLocation({ pageNumber: pageNumber, x: xCoord, y: yCoord })

          if (res && typeof res.then === 'function') {
            await res
          }
          console.log(`Successfully navigated to page ${pageNumber} via viewer.gotoLocation`)
        } catch (err) {
          console.warn('viewer.gotoLocation threw, will try getAPIs fallback', err)
          // Fall through to getAPIs fallback
          throw err
        }
      } else if (typeof (adobeViewer as any).getAPIs === 'function') {
        // Fallback: call apis.gotoLocation if available
        try {
          const apis = await (adobeViewer as any).getAPIs()
          if (apis && typeof apis.gotoLocation === 'function') {
            const res = apis.gotoLocation(pageNumber, xCoord, yCoord)
            if (res && typeof res.then === 'function') await res
            console.log(`Successfully navigated to page ${pageNumber} via apis.gotoLocation`)
          } else {
            console.warn('getAPIs() provided no gotoLocation method')
            throw new Error('No gotoLocation in APIs')
          }
        } catch (err) {
          console.warn('getAPIs().gotoLocation failed', err)
          throw err
        }
      } else {
        console.warn('No known API to perform gotoLocation on this viewer')
        return Promise.reject(new Error('No gotoLocation API'))
      }

      // Optionally search for text after navigation
      if (searchText && (adobeViewer as any).search) {
        try {
          const sres = (adobeViewer as any).search({ query: searchText, matchCase: false, matchWholeWord: false })
          if (sres && typeof sres.then === 'function') await sres
          console.log('Text search completed:', searchText)
        } catch (e) {
          console.warn('Could not search for text after navigation:', e)
        }
      }

      return Promise.resolve()
    } catch (error) {
      console.error('Error navigating to location:', error)
      return Promise.reject(error)
    }
  }

  const enableTextSelection = (enabled: boolean = true) => {
    if (!adobeViewer || !loadedPdfUrl) {
      console.warn('Cannot change text selection state; adobeViewer not ready')
      return
    }

    // Try several known API shapes
    try {
      // Direct method on viewer instance
      if (typeof (adobeViewer as any).enableTextSelection === 'function') {
        try {
          ;(adobeViewer as any).enableTextSelection(enabled)
          console.log(`Called adobeViewer.enableTextSelection(${enabled})`)
          return
        } catch (e) {
          console.warn('adobeViewer.enableTextSelection threw', e)
        }
      }

      // getAPIs fallback
      if (typeof (adobeViewer as any).getAPIs === 'function') {
        ;(adobeViewer as any).getAPIs().then((apis: any) => {
          if (apis && typeof apis.enableTextSelection === 'function') {
            apis.enableTextSelection(enabled)
              .then(() => console.log(`Called apis.enableTextSelection(${enabled})`))
              .catch((err: any) => console.warn('apis.enableTextSelection failed', err))
          } else {
            console.log('No enableTextSelection in apis')
          }
        }).catch((err: any) => {
          console.warn('adobeViewer.getAPIs() failed', err)
        })
        return
      }

      console.log('No known API to enable text selection on this viewer')
    } catch (error) {
      console.error('Error setting text selection:', error)
    }
  }

  const clearPageSelection = (pageNumber: number) => {
    if (!adobeViewer || !loadedPdfUrl) {
      console.warn('Cannot clear page selection; adobeViewer not ready')
      return
    }

    try {
      // Direct API on viewer instance
      if (typeof (adobeViewer as any).clearPageSelection === 'function') {
        try {
          ;(adobeViewer as any).clearPageSelection(pageNumber)
          console.log(`Called adobeViewer.clearPageSelection(${pageNumber})`)
          return
        } catch (e) {
          console.warn('adobeViewer.clearPageSelection threw', e)
        }
      }

      // getAPIs fallback
      if (typeof (adobeViewer as any).getAPIs === 'function') {
        ;(adobeViewer as any).getAPIs().then((apis: any) => {
          if (apis && typeof apis.clearPageSelection === 'function') {
            apis.clearPageSelection(pageNumber)
              .then(() => console.log(`Called apis.clearPageSelection(${pageNumber})`))
              .catch((err: any) => console.warn('apis.clearPageSelection failed', err))
          } else {
            console.log('No clearPageSelection in apis')
          }
        }).catch((err: any) => {
          console.warn('adobeViewer.getAPIs() failed', err)
        })
        return
      }

      console.log('No known API to clear page selection on this viewer')
    } catch (error) {
      console.error('Error clearing page selection:', error)
    }
  }

  // Demo content for fallback
  const demoContent = `Document Insight System - Demo Content

This is a demonstration of the PDF viewer with text selection capabilities. 

Key Features:
• Advanced text selection for AI analysis
• Semantic search across document collections  
• AI-powered insights generation
• Audio overview creation
• Cross-document relationship discovery

Research Methodology:
The system employs state-of-the-art natural language processing techniques to analyze document content. Using transformer-based models, it can identify semantic relationships between different sections of text, even across multiple documents.

Applications:
- Academic research and literature review
- Legal document analysis
- Technical documentation navigation
- Business intelligence and reporting
- Educational content exploration

The cognitive load theory suggests that working memory has limited capacity for processing information. This principle is fundamental to understanding how users interact with complex document systems and interfaces.

To use this system effectively, simply select any text in this document and watch as the AI analyzes your selection against the broader document context.`

  // Adobe PDF Embed view
  if (useAdobeEmbed && ADOBE_CLIENT_ID && apiStatus !== 'error') {
    return (
      <div className="h-full flex flex-col bg-white">
        {/* Header Controls */}
        <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50">
          <div className="flex items-center gap-4">
            
            {documentInfo && (
              <Badge variant="outline" className="text-xs">
                {documentInfo.numPages} pages
              </Badge>
            )}
          </div>

          <div className="flex items-center gap-2">
          </div>
        </div>

        {/* Upload Progress */}
        {isUploading && (
          <div className="p-4 bg-blue-50 border-b border-blue-200">
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <Progress value={uploadProgress} className="h-2" />
              </div>
              <span className="text-sm text-blue-700">{uploadProgress}%</span>
            </div>
            <p className="text-xs text-blue-600 mt-1">Processing document...</p>
          </div>
        )}

        {/* API Status */}
        {apiStatus === 'loading' && (
          <div className="p-4 bg-yellow-50 border-b border-yellow-200">
            <div className="flex items-center gap-2">
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-yellow-600 border-t-transparent" />
              <span className="text-sm text-yellow-700">Loading Adobe PDF Embed API...</span>
            </div>
          </div>
        )}

        {/* PDF Viewer Container */}
        <div className="flex-1 relative bg-slate-100">
          {isLoading && (
            <div className="absolute inset-0 bg-white bg-opacity-90 flex items-center justify-center z-10">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-600 border-t-transparent mx-auto mb-4" />
                <p className="text-slate-600">Loading PDF...</p>
              </div>
            </div>
          )}

          {!loadedPdfUrl && !isLoading && (
            <div className="flex items-center justify-center h-full">
              <Card className="w-96">
                <CardContent className="pt-6">
                  <div className="text-center">
                    <FileText className="w-16 h-16 text-slate-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-slate-900 mb-2">No PDF Loaded</h3>
                    <p className="text-slate-600 mb-4">Upload a PDF file to start analyzing</p>
                    <Button onClick={() => fileInputRef.current?.click()} className="w-full">
                      <Upload className="w-4 h-4 mr-2" />
                      Choose PDF File
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Adobe PDF Embed Container */}
          <div 
            id="adobe-dc-view-container"
            ref={viewerContainerRef}
            className="w-full h-full"
            style={{ 
              display: loadedPdfUrl && !isLoading ? 'block' : 'none',
              minHeight: '500px'
            }}
          />
        </div>

        {/* Text Selection Display */}
        {selectedText && (
          <div className="bg-blue-50 border-t border-blue-200 p-4">
            <div className="flex items-start gap-3">
              <Search className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-medium text-blue-900">Selected Text:</span>
                  <Badge variant="secondary" className="text-xs">
                    {selectedText.length} chars
                  </Badge>
                </div>
                <p className="text-sm text-blue-800 italic break-words">
                  "{selectedText.substring(0, 200)}{selectedText.length > 200 ? '...' : ''}"
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // Fallback demo mode
  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50">
        <div className="flex items-center gap-4">
        </div>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">
            <ZoomIn className="w-4 h-4" />
          </Button>
          <Button variant="outline" size="sm">
            <ZoomOut className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Adobe API Notice */}
      <Alert className="m-4 border-amber-200 bg-amber-50">
        <AlertCircle className="h-4 w-4 text-amber-600" />
        <AlertDescription className="text-amber-800">
          <strong>Text Selection:</strong> {useAdobeEmbed && apiStatus === 'ready' 
            ? "Adobe PDF Embed enabled - select text directly in the PDF viewer below."
            : "For text selection, copy text from the PDF and paste it in the text area below, or use the manual input field."}
        </AlertDescription>
      </Alert>

      {/* Manual Text Input for Analysis */}
      {!useAdobeEmbed || apiStatus !== 'ready' ? (
        <div className="m-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <Search className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-900">Manual Text Selection</span>
          </div>
          {lastSelectedText && (
            <div className="mb-3 p-2 bg-green-50 border border-green-200 rounded text-sm text-green-800">
              <strong>✓ Last analyzed:</strong> {lastSelectedText}
            </div>
          )}
          <textarea
            className="w-full h-24 p-3 border border-blue-300 rounded-md resize-none text-sm"
            placeholder="Copy and paste text from the PDF above to analyze for insights and connections..."
            value={manualText}
            onChange={(e) => setManualText(e.target.value)}
          />
          <div className="flex gap-2 mt-2">
            <Button
              onClick={() => {
                if (manualText.trim().length > 10) {
                  setLastSelectedText(manualText.substring(0, 50) + '...')
                  onTextSelection(manualText.trim())
                  setManualText('')
                }
              }}
              disabled={manualText.trim().length < 10}
              size="sm"
            >
              Analyze Text
            </Button>
            <Button
              onClick={() => {
                const testText = "Transfer learning is a machine learning technique where a model developed for a particular task is reused as the starting point for a model on a second related task. This approach is particularly valuable in deep learning because training deep neural networks requires very large amounts of data, and the availability of labeled training data can be limited."
                setLastSelectedText(testText.substring(0, 50) + '...')
                onTextSelection(testText)
                console.log('Test text selection triggered:', testText)
              }}
              variant="outline"
              size="sm"
            >
              Test with Sample Text
            </Button>
          </div>
        </div>
      ) : null}

      {/* PDF Display */}
      <div className="flex-1 overflow-auto bg-slate-100 p-4">
        {loadedPdfUrl ? (
          <div className="max-w-full mx-auto">
            <Card className="shadow-lg">
              <CardContent className="p-0">
                <div className="relative">
                  <iframe
                    key={loadedPdfUrl + Date.now()} // Force remount when URL changes
                    src={`${loadedPdfUrl}#toolbar=1&navpanes=1&scrollbar=1&nocache=${Date.now()}`}
                    className="w-full h-[800px] border-0"
                    title="PDF Viewer"
                    allow="fullscreen"
                    onLoad={() => {
                      console.log('PDF loaded in iframe:', loadedPdfUrl)
                    }}
                    onError={(e) => {
                      console.error('Error loading PDF in iframe:', e)
                    }}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        ) : (
          <Card className="max-w-4xl mx-auto">
            <CardContent className="p-8 text-center">
              <FileText className="w-16 h-16 text-slate-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-slate-900 mb-2">No PDF Loaded</h3>
              <p className="text-slate-600 mb-4">Upload a PDF file to start analyzing</p>
              <Button onClick={() => fileInputRef.current?.click()} className="w-auto">
                <Upload className="w-4 h-4 mr-2" />
                Choose PDF File
              </Button>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Selection Display */}
      {selectedText && (
        <div className="bg-blue-50 border-t border-blue-200 p-4">
          <div className="flex items-start gap-3">
            <Search className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-medium text-blue-900">Selected Text:</span>
                <Badge variant="secondary" className="text-xs">
                  {selectedText.length} chars
                </Badge>
              </div>
              <p className="text-sm text-blue-800 italic break-words">
                "{selectedText.substring(0, 200)}{selectedText.length > 200 ? '...' : ''}"
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
})

PDFViewer.displayName = 'PDFViewer'

export default PDFViewer

// Export the goToLocation function for external use
export const usePDFNavigation = () => {
  return {
    goToLocation: (pageNumber: number, searchText?: string) => {
      // This would be connected to the PDF viewer instance
      console.log(`Navigate to page ${pageNumber}`, searchText)
    }
  }
}
