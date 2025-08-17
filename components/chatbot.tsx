import { useState } from "react"
import { SendHorizonal, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

interface ChatbotProps {
  documentId?: string // Optionally pass a document ID for context
}

export default function Chatbot({ documentId }: ChatbotProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)

  const sendMessage = async () => {
    if (!input.trim()) return
    setLoading(true)
    setMessages([...messages, { role: "user", content: input }])
    try {
      const response = await fetch("/api/qa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: input,
          document_id: documentId,
        }),
      })
      if (response.ok) {
        const data = await response.json()
        setMessages((prev) => [...prev, { role: "assistant", content: data.answer }])
      } else {
        setMessages((prev) => [...prev, { role: "assistant", content: "Sorry, I couldn't answer that." }])
      }
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: "Error connecting to AI service." }])
    } finally {
      setLoading(false)
      setInput("")
    }
  }

  return (
    <div className="h-full flex flex-col bg-slate-50">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 min-h-0">
        {messages.length === 0 ? (
          <div className="text-xs text-slate-500 text-center py-8">
            Ask questions about the current document...
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`text-sm ${msg.role === "user" ? "text-blue-900" : "text-green-900"}`}>
              <span className="font-semibold">{msg.role === "user" ? "You:" : "AI:"}</span>
              <div className="mt-1 text-slate-700">{msg.content}</div>
            </div>
          ))
        )}
      </div>
      
      {/* Input area - sticky to bottom */}
      <div className="border-t border-slate-200 p-4 bg-white">
        <div className="flex gap-2">
          <input
            className="flex-1 border border-slate-300 rounded-md px-3 py-2 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter") sendMessage() }}
            placeholder="Ask a question about this document..."
            disabled={loading}
          />
          <Button 
            size="sm" 
            onClick={sendMessage} 
            disabled={loading || !input.trim()}
            className="px-3"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <SendHorizonal className="w-4 h-4" />}
          </Button>
        </div>
      </div>
    </div>
  )
}
