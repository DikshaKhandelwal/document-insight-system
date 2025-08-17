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
    <div className="bg-slate-50 border rounded-lg p-4 flex flex-col gap-3">
      <div className="flex-1 overflow-y-auto max-h-64 mb-2">
        {messages.map((msg, idx) => (
          <div key={idx} className={`mb-2 text-sm ${msg.role === "user" ? "text-blue-900" : "text-green-900"}`}>
            <span className="font-semibold mr-2">{msg.role === "user" ? "You:" : "AI:"}</span>
            {msg.content}
          </div>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          className="flex-1 border rounded px-2 py-1 text-sm"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") sendMessage() }}
          placeholder="Ask a question about this document..."
          disabled={loading}
        />
        <Button size="sm" onClick={sendMessage} disabled={loading || !input.trim()}>
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <SendHorizonal className="w-4 h-4" />}
        </Button>
      </div>
    </div>
  )
}
