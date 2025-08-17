"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ExternalLink, BookOpen, AlertTriangle, Lightbulb, Link } from "lucide-react"
import { cn } from "@/lib/utils"

interface SnippetCardProps {
  snippet: {
    id: string
    documentName: string
    heading: string
    snippet: string
    page: number
    confidence: number
  }
  onClick: () => void
  type: "related" | "overlapping" | "contradicting" | "examples"
}

const typeConfig = {
  related: {
    icon: Link,
    color: "blue",
    bgColor: "bg-blue-50",
    borderColor: "border-blue-200",
    textColor: "text-blue-700",
  },
  overlapping: {
    icon: BookOpen,
    color: "green",
    bgColor: "bg-green-50",
    borderColor: "border-green-200",
    textColor: "text-green-700",
  },
  contradicting: {
    icon: AlertTriangle,
    color: "red",
    bgColor: "bg-red-50",
    borderColor: "border-red-200",
    textColor: "text-red-700",
  },
  examples: {
    icon: Lightbulb,
    color: "amber",
    bgColor: "bg-amber-50",
    borderColor: "border-amber-200",
    textColor: "text-amber-700",
  },
}

export default function SnippetCard({ snippet, onClick, type }: SnippetCardProps) {
  const config = typeConfig[type]
  const IconComponent = config.icon

  return (
    <Card className={cn("insight-card p-4 cursor-pointer border-l-4", config.borderColor)}>
      <div className="flex items-start gap-3 mb-3">
        <div className={cn("p-2 rounded-lg", config.bgColor)}>
          <IconComponent className={cn("w-4 h-4", config.textColor)} />
        </div>
        <div className="flex-1">
          <div
            className="font-medium text-slate-800 text-[0.95rem] mb-1 break-all whitespace-normal w-full"
            title={snippet.documentName}
          >
            {snippet.documentName}
          </div>
          <p className="text-xs text-slate-500">
            {snippet.heading} • Page {snippet.page}
          </p>
        </div>
        <div className="text-xs text-slate-400">{Math.round(snippet.confidence * 100)}%</div>
      </div>

      <p className="text-sm text-slate-700 leading-relaxed mb-4">{snippet.snippet}</p>

      <Button onClick={onClick} variant="outline" size="sm" className="w-full text-xs bg-transparent">
        <ExternalLink className="w-3 h-3 mr-2" />
        View in Document
      </Button>
    </Card>
  )
}
