"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Upload, FileText, Sparkles, Brain, Search, Zap, ArrowRight, BookOpen, Target, Lightbulb } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import UploadArea from "@/components/upload-area"
import HeaderBar from "@/components/header-bar"

export default function HomePage() {
  const [pastDocuments, setPastDocuments] = useState<File[]>([])
  const [currentDocument, setCurrentDocument] = useState<File | null>(null)
  const router = useRouter()

  const handlePastDocumentsUpload = (files: File[]) => {
    setPastDocuments((prev) => [...prev, ...files])
  }

  const handleCurrentDocumentUpload = (files: File[]) => {
    if (files.length > 0) {
      setCurrentDocument(files[0])
      // Navigate to reader page
      router.push("/reader")
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-red-50">
      <HeaderBar />

      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-gradient-to-br from-red-900/10 via-slate-600/5 to-red-800/8"></div>
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(185,28,28,0.12),transparent_50%)]"></div>
        </div>

        <div className="relative max-w-7xl mx-auto px-6 pt-12 pb-20">
          <div className="text-center mb-20">
            <div className="inline-flex items-center gap-2 bg-white/80 backdrop-blur-sm rounded-full px-4 py-2 mb-8 border border-slate-200">
              <Sparkles className="w-4 h-4 text-red-700" />
              <span className="text-sm font-medium text-slate-700">AI-Powered Document Intelligence</span>
            </div>

            <h1 className="font-serif text-4xl md:text-6xl lg:text-7xl font-bold text-slate-900 mb-8 leading-tight">
              Document{" "}
              <span className="relative">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-800 to-slate-700">
                  Insight
                </span>
                <div className="absolute -inset-1 bg-gradient-to-r from-red-800/20 to-slate-700/20 blur-lg -z-10"></div>
              </span>
            </h1>

            <p className="text-xl md:text-2xl text-slate-600 max-w-4xl mx-auto leading-relaxed font-light">
              Transform how you read and understand PDFs. Discover hidden connections, contradictions, and insights
              across your entire document library with AI precision.
            </p>

            <div className="flex flex-wrap justify-center gap-4 mt-8">
              <div className="flex items-center gap-2 bg-white/60 backdrop-blur-sm rounded-full px-4 py-2 border border-slate-200">
                <Target className="w-4 h-4 text-red-700" />
                <span className="text-sm font-medium text-slate-700">Semantic Search</span>
              </div>
              <div className="flex items-center gap-2 bg-white/60 backdrop-blur-sm rounded-full px-4 py-2 border border-slate-200">
                <Lightbulb className="w-4 h-4 text-red-700" />
                <span className="text-sm font-medium text-slate-700">Smart Insights</span>
              </div>
              <div className="flex items-center gap-2 bg-white/60 backdrop-blur-sm rounded-full px-4 py-2 border border-slate-200">
                <BookOpen className="w-4 h-4 text-red-700" />
                <span className="text-sm font-medium text-slate-700">Interactive Reading</span>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mb-20">
            <Card className="group relative overflow-hidden bg-white/80 backdrop-blur-sm border-0 shadow-lg hover:shadow-xl transition-all duration-300 p-8 text-center">
              <div className="absolute inset-0 bg-gradient-to-br from-red-900/10 via-slate-600/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-slate-700 flex items-center justify-center shadow-lg border border-slate-700/20">
                  <Brain className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-serif text-xl font-semibold mb-4 text-slate-900">Semantic Understanding</h3>
                <p className="text-slate-600 leading-relaxed">
                  AI analyzes meaning, not just keywords, to find truly relevant connections across documents.
                </p>
              </div>
            </Card>

            <Card className="group relative overflow-hidden bg-white/80 backdrop-blur-sm border-0 shadow-lg hover:shadow-xl transition-all duration-300 p-8 text-center">
              <div className="absolute inset-0 bg-gradient-to-br from-red-900/10 via-slate-600/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-slate-700 flex items-center justify-center shadow-lg border border-slate-700/20">
                  <Search className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-serif text-xl font-semibold mb-4 text-slate-900">Instant Discovery</h3>
                <p className="text-slate-600 leading-relaxed">
                  Select any text and instantly see related, overlapping, and contradicting sections from your library.
                </p>
              </div>
            </Card>

            <Card className="group relative overflow-hidden bg-white/80 backdrop-blur-sm border-0 shadow-lg hover:shadow-xl transition-all duration-300 p-8 text-center">
              <div className="absolute inset-0 bg-gradient-to-br from-red-900/10 via-slate-600/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-slate-700 flex items-center justify-center shadow-lg border border-slate-700/20">
                  <Zap className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-serif text-xl font-semibold mb-4 text-slate-900">Smart Insights</h3>
                <p className="text-slate-600 leading-relaxed">
                  Generate comprehensive insights and audio overviews powered by advanced language models.
                </p>
              </div>
            </Card>
          </div>

          <div className="grid lg:grid-cols-2 gap-12">
            <div className="space-y-6">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-xl bg-slate-700 flex items-center justify-center">
                  <FileText className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="font-serif text-2xl font-semibold text-slate-900">Build Your Library</h2>
                  <p className="text-slate-600">Upload existing PDFs to create your knowledge base</p>
                </div>
              </div>

              <UploadArea
                title="Past Documents"
                description="Drag & drop multiple PDFs or click to browse"
                multiple={true}
                onUpload={handlePastDocumentsUpload}
                uploadedFiles={pastDocuments}
                icon={<Upload className="w-6 h-6" />}
                className="h-72"
              />

              {pastDocuments.length > 0 && (
                <div className="flex items-center gap-2 text-sm text-slate-600 bg-white/60 backdrop-blur-sm rounded-lg px-4 py-2 border border-slate-200">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  {pastDocuments.length} document{pastDocuments.length !== 1 ? "s" : ""} ready for analysis
                </div>
              )}
            </div>

            <div className="space-y-6">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-xl bg-slate-700 flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="font-serif text-2xl font-semibold text-slate-900">Start Reading</h2>
                  <p className="text-slate-600">Open a PDF to begin intelligent analysis</p>
                </div>
              </div>

              <UploadArea
                title="Current Document"
                description="Upload the PDF you want to read and analyze"
                multiple={false}
                onUpload={handleCurrentDocumentUpload}
                uploadedFiles={currentDocument ? [currentDocument] : []}
                icon={<FileText className="w-6 h-6" />}
                className="h-72"
                primary={true}
              />

              {currentDocument && (
                <Button
                  onClick={() => router.push("/reader")}
                  className="w-full bg-slate-700 text-white font-medium py-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 group"
                >
                  Open in Reader
                  <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform duration-200" />
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="relative bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-24 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_30%,rgba(185,28,28,0.15),transparent_70%)]"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_80%,rgba(100,116,139,0.1),transparent_70%)]"></div>

        <div className="relative max-w-5xl mx-auto px-6 text-center">
          <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-sm rounded-full px-4 py-2 mb-8 border border-white/20">
            <BookOpen className="w-4 h-4 text-red-400" />
            <span className="text-sm font-medium text-white/90">The Future of Document Reading</span>
          </div>

          <h2 className="font-serif text-3xl md:text-4xl font-bold text-white mb-12">Every PDF Has a Story to Tell</h2>

          <div className="grid md:grid-cols-2 gap-12 text-left">
            <div className="space-y-6">
              <p className="text-lg text-slate-300 leading-relaxed">
                Every PDF tells a story, but that story is often fragmented across pages, buried in dense text, or
                scattered across multiple documents. Our system reads between the lines, connecting ideas that span
                chapters and finding contradictions that need resolution.
              </p>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-red-400 rounded-full mt-3 flex-shrink-0"></div>
                <p className="text-slate-300">
                  <strong className="text-white">For Researchers:</strong> Analyze academic papers with unprecedented
                  depth
                </p>
              </div>
            </div>

            <div className="space-y-6">
              <p className="text-lg text-slate-300 leading-relaxed">
                Whether you're a lawyer reviewing case documents or a student studying complex materials, Document
                Insight transforms passive reading into active discovery, revealing insights that would take hours to
                find manually.
              </p>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-red-400 rounded-full mt-3 flex-shrink-0"></div>
                <p className="text-slate-300">
                  <strong className="text-white">For Professionals:</strong> Navigate complex documents with AI guidance
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}