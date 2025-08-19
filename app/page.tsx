"use client"

import { useState, useEffect } from "react"
import { Upload, FileText, Sparkles, Brain, Search, Zap, ArrowRight, BookOpen, Target, Lightbulb, Star, Users, Shield, Globe } from "lucide-react"

export default function HomePage() {
  const [scrollY, setScrollY] = useState(0)
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY)
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY })
    }
    
    window.addEventListener('scroll', handleScroll)
    window.addEventListener('mousemove', handleMouseMove)
    
    return () => {
      window.removeEventListener('scroll', handleScroll)
      window.removeEventListener('mousemove', handleMouseMove)
    }
  }, [])

  const features = [
    {
      icon: Brain,
      title: "Neural Understanding",
      description: "Advanced AI that comprehends context, meaning, and relationships across your entire document library.",
      gradient: "from-purple-400 to-indigo-500"
    },
    {
      icon: Search,
      title: "Quantum Search",
      description: "Find connections that don't exist in traditional search. Discover insights hidden between the lines.",
      gradient: "from-blue-400 to-cyan-400"
    },
    {
      icon: Zap,
      title: "Instant Synthesis",
      description: "Generate comprehensive insights, summaries, and audio content at the speed of thought.",
      gradient: "from-amber-400 to-orange-500"
    }
  ]

  return (
    <div className="min-h-screen bg-white text-gray-900 overflow-hidden">
      {/* Dynamic Background */}
      <div className="fixed inset-0 z-0">
        <div 
          className="absolute inset-0 opacity-20"
          style={{
            background: `radial-gradient(circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(251, 191, 36, 0.1) 0%, transparent 50%)`
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-br from-slate-50 via-white to-blue-50" />
        <div className="absolute inset-0">
          {[...Array(30)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-amber-300 rounded-full opacity-30 animate-pulse"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 3}s`,
                animationDuration: `${2 + Math.random() * 3}s`
              }}
            />
          ))}
        </div>
      </div>

      {/* Header */}
      <header className="relative z-50 backdrop-blur-xl bg-white/80 border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center shadow-lg">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
              DocuVerse
            </span>
          </div>
          <nav className="hidden md:flex items-center space-x-8">
            <a href="#features" className="text-gray-600 hover:text-gray-900 transition-all duration-300 hover:scale-105">Features</a>
            <a href="#how-it-works" className="text-gray-600 hover:text-gray-900 transition-all duration-300 hover:scale-105">How it Works</a>
            <a href="#pricing" className="text-gray-600 hover:text-gray-900 transition-all duration-300 hover:scale-105">Pricing</a>
            <a href="/reader">
              <button className="bg-gradient-to-r from-amber-400 to-orange-500 text-white px-6 py-2 rounded-full font-medium hover:shadow-lg hover:shadow-amber-400/30 transition-all duration-300 hover:scale-105 shadow-md">
                Get Started
              </button>
            </a>
          </nav>
        </div>
      </header>

      <div className="relative z-10">
        {/* Hero Section */}
        <section className="relative min-h-screen flex items-center justify-center px-6">
          <div className="max-w-7xl mx-auto text-center">

            {/* Main Headline */}
            <h1 className="text-6xl md:text-8xl font-black mb-8 leading-tight">
              <span className="block text-4xl md:text-6xl bg-gradient-to-r from-gray-800 via-gray-700 to-gray-600 bg-clip-text text-transparent mb-2">
                Transform
              </span>
              <span className="block text-4xl md:text-6xl bg-gradient-to-r from-amber-500 via-orange-500 to-red-500 bg-clip-text text-transparent">
                Reading
              </span>
              <span className="block text-4xl md:text-6xl bg-gradient-to-r from-gray-800 via-gray-700 to-gray-600 bg-clip-text text-transparent">
                Forever
              </span>
            </h1>
            <div className="text-2xl md:text-3xl text-amber-600 font-semibold mb-6">
              Read. Discover. Connect.
            </div>

            <p className="text-xl md:text-2xl text-gray-600 max-w-4xl mx-auto leading-relaxed mb-12 font-light">
              Experience the future of document intelligence. Our AI doesn't just read—it understands, connects, and reveals insights that traditional tools miss entirely.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-6 mb-16">
              <a href="/reader">
                <button className="group relative overflow-hidden bg-gradient-to-r from-amber-400 via-orange-500 to-red-500 text-white font-bold px-12 py-4 rounded-2xl shadow-2xl hover:shadow-amber-400/40 transition-all duration-500 transform hover:scale-105 hover:-translate-y-1">
                  <div className="absolute inset-0 bg-gradient-to-r from-amber-500 via-orange-600 to-red-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  <div className="relative flex items-center gap-3 text-lg">
                    <BookOpen className="w-6 h-6" />
                    Launch DocuVerse
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-300" />
                  </div>
                </button>
              </a>
              <button className="group border-2 border-gray-300 text-gray-700 font-medium px-8 py-4 rounded-2xl hover:bg-gray-50 transition-all duration-300 hover:border-gray-400 hover:scale-105 shadow-md">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full bg-green-400 animate-pulse" />
                  Watch Demo
                </div>
              </button>
            </div>

          </div>

          {/* Floating Elements */}
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            <div className="absolute top-1/4 left-10 w-32 h-32 bg-gradient-to-br from-amber-300/20 to-orange-300/20 rounded-full blur-xl animate-pulse" />
            <div className="absolute bottom-1/3 right-16 w-48 h-48 bg-gradient-to-br from-blue-300/15 to-purple-300/15 rounded-full blur-2xl animate-pulse" />
            <div className="absolute top-1/2 right-1/4 w-24 h-24 bg-gradient-to-br from-red-300/20 to-pink-300/20 rounded-full blur-lg animate-pulse" />
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="relative py-24 px-6 bg-gradient-to-br from-gray-50 to-blue-50">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-20">
              <h2 className="text-5xl md:text-6xl font-bold mb-6">
                <span className="bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                  Beyond Traditional
                </span>
                <br />
                <span className="bg-gradient-to-r from-amber-500 to-orange-500 bg-clip-text text-transparent">
                  Document Reading
                </span>
              </h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Our AI doesn't just process text—it understands meaning, context, and relationships in ways that feel almost magical.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {features.map((feature, index) => (
                <div key={index} className="group relative">
                  <div className="absolute inset-0 bg-white/60 rounded-3xl blur-xl group-hover:blur-2xl transition-all duration-500" />
                  <div className="relative bg-white/80 backdrop-blur-xl rounded-3xl p-8 border border-gray-200 hover:border-gray-300 transition-all duration-500 hover:scale-105 hover:-translate-y-2 shadow-lg hover:shadow-xl">
                    <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg`}>
                      <feature.icon className="w-8 h-8 text-white" />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-800 mb-4">{feature.title}</h3>
                    <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                    <div className="mt-6 flex items-center text-amber-600 group-hover:text-amber-700 transition-colors duration-300">
                      <span className="text-sm font-medium">Learn more</span>
                      <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform duration-300" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section id="how-it-works" className="relative py-24 px-6 bg-white">
          <div className="max-w-6xl mx-auto text-center">
            <h2 className="text-4xl md:text-5xl font-bold mb-16">
              <span className="bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                Four Steps to
              </span>
              <br />
              <span className="bg-gradient-to-r from-amber-500 to-orange-500 bg-clip-text text-transparent">
                Document Mastery
              </span>
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              {[
                { icon: Upload, title: "Upload", description: "Drag & drop your documents into our secure AI environment" },
                { icon: Brain, title: "Process", description: "Our neural networks analyze and understand your content" },
                { icon: Search, title: "Discover", description: "Find hidden connections and insights across all documents" },
                { icon: Zap, title: "Transform", description: "Generate summaries, podcasts, and interactive experiences" }
              ].map((step, index) => (
                <div key={index} className="relative group">
                  <div className="flex flex-col items-center">
                    <div className="relative mb-6">
                      <div className="w-20 h-20 rounded-full bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center group-hover:scale-110 transition-transform duration-300 shadow-lg">
                        <step.icon className="w-10 h-10 text-white" />
                      </div>
                      <div className="absolute -top-2 -right-2 w-8 h-8 bg-gray-800 rounded-full flex items-center justify-center text-white font-bold text-sm shadow-lg">
                        {index + 1}
                      </div>
                    </div>
                    <h3 className="text-xl font-bold text-gray-800 mb-3">{step.title}</h3>
                    <p className="text-gray-600 text-center">{step.description}</p>
                  </div>
                  {index < 3 && (
                    <div className="hidden md:block absolute top-10 left-full w-full h-0.5 bg-gradient-to-r from-amber-400 to-transparent" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="relative py-24 px-6 bg-gradient-to-br from-amber-50 to-orange-50">
          <div className="max-w-4xl mx-auto text-center">
            <div className="relative bg-white/70 backdrop-blur-xl rounded-3xl p-12 border border-gray-200 shadow-2xl">
              <div className="absolute inset-0 bg-gradient-to-br from-amber-100/30 to-orange-100/30 rounded-3xl" />
              <div className="relative">
                <h2 className="text-4xl md:text-5xl font-bold mb-6">
                  <span className="bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                    Ready to Experience
                  </span>
                  <br />
                  <span className="bg-gradient-to-r from-amber-500 to-orange-500 bg-clip-text text-transparent">
                    The Future?
                  </span>
                </h2>
                <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
                  Join thousands of professionals who've transformed their document workflow with DocuVerse.
                </p>
                <a href="/reader">
                  <button className="group relative overflow-hidden bg-gradient-to-r from-amber-400 via-orange-500 to-red-500 text-white font-bold px-12 py-4 rounded-2xl shadow-2xl hover:shadow-amber-400/40 transition-all duration-500 transform hover:scale-105 hover:-translate-y-1">
                    <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    <div className="relative flex items-center gap-3 text-lg">
                      <Sparkles className="w-6 h-6 animate-spin" />
                      Start Your Journey
                      <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-300" />
                    </div>
                  </button>
                </a>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="relative py-12 px-6 border-t border-gray-200 bg-gray-50">
          <div className="max-w-7xl mx-auto">
            <div className="flex flex-col md:flex-row items-center justify-between">
              <div className="flex items-center space-x-3 mb-6 md:mb-0">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center shadow-lg">
                  <BookOpen className="w-6 h-6 text-white" />
                </div>
                <span className="text-2xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                  DocuVerse
                </span>
              </div>
              <div className="flex items-center space-x-8">
                <a href="#" className="text-gray-600 hover:text-gray-800 transition-colors duration-300">Privacy</a>
                <a href="#" className="text-gray-600 hover:text-gray-800 transition-colors duration-300">Terms</a>
                <a href="#" className="text-gray-600 hover:text-gray-800 transition-colors duration-300">Support</a>
                <div className="flex items-center gap-2 text-gray-600">
                  <Globe className="w-4 h-4" />
                  <span className="text-sm">© 2025 DocuVerse AI</span>
                </div>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}