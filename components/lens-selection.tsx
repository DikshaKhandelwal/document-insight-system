"use client"

import React from 'react'
import { Search, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface LensSelectionProps {
  isActive: boolean
  currentSelection: {
    start: { x: number; y: number }
    end: { x: number; y: number }
  } | null
  savedSelections: Array<{
    id: string
    text: string
    bbox: { x: number; y: number; width: number; height: number }
  }>
  onClearSelection: (id: string) => void
  onAnalyzeSelection: (text: string) => void
}

export default function LensSelection({
  isActive,
  currentSelection,
  savedSelections,
  onClearSelection,
  onAnalyzeSelection,
}: LensSelectionProps) {
  if (!isActive) return null

  return (
    <div className="absolute inset-0 pointer-events-none z-10">
      {/* Current selection box */}
      {currentSelection && (
        <div
          className="absolute border-2 border-blue-500 bg-blue-500/20 rounded-sm shadow-lg"
          style={{
            left: Math.min(currentSelection.start.x, currentSelection.end.x),
            top: Math.min(currentSelection.start.y, currentSelection.end.y),
            width: Math.abs(currentSelection.end.x - currentSelection.start.x),
            height: Math.abs(currentSelection.end.y - currentSelection.start.y),
          }}
        >
          <div className="absolute -top-1 -left-1 w-3 h-3 bg-blue-500 rounded-full border-2 border-white shadow-sm" />
          <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-blue-500 rounded-full border-2 border-white shadow-sm" />
        </div>
      )}
      
      {/* Saved selection boxes */}
      {savedSelections.map((selection) => (
        <div
          key={selection.id}
          className="absolute border-2 border-green-500 bg-green-500/10 rounded-sm group shadow-md hover:shadow-lg transition-shadow"
          style={{
            left: selection.bbox.x,
            top: selection.bbox.y,
            width: selection.bbox.width,
            height: selection.bbox.height,
          }}
        >
          {/* Selection corners */}
          <div className="absolute -top-1 -left-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white shadow-sm" />
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white shadow-sm" />
          <div className="absolute -bottom-1 -left-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white shadow-sm" />
          <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white shadow-sm" />
          
          {/* Tooltip */}
          <div className="absolute -top-12 left-0 bg-green-600 text-white text-xs px-3 py-1.5 rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity max-w-xs z-20 pointer-events-auto">
            <div className="font-medium mb-1">Selected Text</div>
            <div className="text-green-100 mb-2">
              {selection.text.substring(0, 80)}
              {selection.text.length > 80 ? '...' : ''}
            </div>
            <div className="flex gap-1">
              <Button
                onClick={() => onAnalyzeSelection(selection.text)}
                size="sm"
                variant="ghost"
                className="h-5 px-2 text-green-100 hover:text-white hover:bg-green-700"
              >
                <Search className="w-3 h-3 mr-1" />
                Analyze
              </Button>
              <Button
                onClick={() => onClearSelection(selection.id)}
                size="sm"
                variant="ghost"
                className="h-5 px-2 text-green-100 hover:text-white hover:bg-red-600"
              >
                <X className="w-3 h-3" />
              </Button>
            </div>
            {/* Arrow pointing down */}
            <div className="absolute top-full left-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-l-transparent border-r-transparent border-t-green-600" />
          </div>
        </div>
      ))}
      
      {/* Instructions overlay */}
      {savedSelections.length === 0 && !currentSelection && (
        <div className="absolute top-8 left-1/2 transform -translate-x-1/2 bg-blue-600 text-white px-6 py-3 rounded-lg shadow-lg flex items-center gap-3 z-20 animate-pulse">
          <div className="flex items-center gap-2">
            <Search className="w-5 h-5" />
            <span className="font-medium">Lens Mode Active</span>
          </div>
          <div className="text-blue-100 text-sm">
            Click and drag to select text regions for AI analysis
          </div>
        </div>
      )}
      
      {/* Google Lens-style scanning animation */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="lens-scanner" />
      </div>
      
      <style jsx>{`
        .lens-scanner {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 2px;
          background: linear-gradient(90deg, transparent, #3b82f6, transparent);
          animation: scan 3s ease-in-out infinite;
        }
        
        @keyframes scan {
          0% {
            top: 0;
            opacity: 0.7;
          }
          50% {
            opacity: 0.3;
          }
          100% {
            top: 100%;
            opacity: 0.7;
          }
        }
      `}</style>
    </div>
  )
}
