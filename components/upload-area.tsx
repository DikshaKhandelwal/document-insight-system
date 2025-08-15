"use client"

import type React from "react"
import { useState, useCallback, useRef } from "react"
import { Upload, File, X } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface UploadAreaProps {
  title: string
  description: string
  multiple: boolean
  onUpload: (files: File[]) => void
  uploadedFiles: File[]
  icon: React.ReactNode
  className?: string
  primary?: boolean
}

export default function UploadArea({
  title,
  description,
  multiple,
  onUpload,
  uploadedFiles,
  icon,
  className,
  primary = false,
}: UploadAreaProps) {
  const [isDragActive, setIsDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragActive(false)

      const allFiles = Array.from(e.dataTransfer.files)
      const pdfFiles = allFiles.filter((file) => file.type === "application/pdf")

      if (allFiles.length > pdfFiles.length) {
        console.log("[v0] Some files were filtered out - only PDFs are supported")
      }

      if (pdfFiles.length > 0) {
        if (multiple) {
          onUpload(pdfFiles)
          console.log("[v0] Uploaded", pdfFiles.length, "PDF files")
        } else {
          onUpload([pdfFiles[0]])
          console.log("[v0] Uploaded PDF:", pdfFiles[0].name)
        }
      }
    },
    [multiple, onUpload],
  )

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || [])
      if (files.length > 0) {
        onUpload(files)
      }
    },
    [onUpload],
  )

  const handleClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const removeFile = (index: number) => {
    const newFiles = uploadedFiles.filter((_, i) => i !== index)
    onUpload(newFiles)
  }

  return (
    <div className="space-y-4">
      <Card
        className={cn(
          "relative overflow-hidden cursor-pointer transition-all duration-300 border-2 border-dashed bg-white/60 backdrop-blur-sm hover:bg-white/80",
          isDragActive
            ? "border-blue-400 bg-blue-50/80 scale-[1.02]"
            : primary
              ? "border-blue-300 hover:border-blue-400"
              : "border-slate-300 hover:border-slate-400",
          className,
        )}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          multiple={multiple}
          onChange={handleFileSelect}
          className="hidden"
        />

        <div className="flex flex-col items-center justify-center p-8 text-center">
          <div
            className={cn(
              "w-16 h-16 rounded-2xl flex items-center justify-center mb-6 shadow-lg transition-all duration-300",
              primary ? "adobe-gradient text-white" : "bg-gradient-to-br from-slate-500 to-slate-600 text-white",
              isDragActive && "scale-110",
            )}
          >
            {icon}
          </div>
          <h3 className="font-serif text-xl font-semibold mb-3 text-slate-900">{title}</h3>
          <p className="text-slate-600 mb-6 leading-relaxed">{description}</p>
          <Button
            variant="outline"
            className={cn(
              "bg-white/80 backdrop-blur-sm border-slate-300 hover:bg-white transition-all duration-200",
              primary && "border-blue-300 hover:border-blue-400",
            )}
          >
            <Upload className="w-4 h-4 mr-2" />
            Choose Files
          </Button>
        </div>

        {isDragActive && (
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-blue-600/10 flex items-center justify-center">
            <div className="text-blue-600 font-medium">Drop files here</div>
          </div>
        )}
      </Card>

      {uploadedFiles.length > 0 && (
        <div className="space-y-3">
          {uploadedFiles.map((file, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-4 bg-white/80 backdrop-blur-sm rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-all duration-200"
            >
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-lg bg-red-100 flex items-center justify-center">
                  <File className="w-5 h-5 text-red-600" />
                </div>
                <div>
                  <span className="font-medium text-slate-900 block truncate max-w-xs">{file.name}</span>
                  <span className="text-sm text-slate-500">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation()
                  removeFile(index)
                }}
                className="text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
