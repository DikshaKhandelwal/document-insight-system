import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

export async function GET(request: NextRequest) {
  try {
    const res = await fetch(`${BACKEND_URL}/audio-list`)
    if (!res.ok) {
      console.error('Backend /audio-list returned', res.status)
      return NextResponse.json({ filename: null })
    }
    const data = await res.json()
    if (Array.isArray(data) && data.length > 0) {
      return NextResponse.json({ filename: data[0] })
    }
    return NextResponse.json({ filename: null })
  } catch (err) {
    console.error('Failed to fetch latest audio from backend', err)
    return NextResponse.json({ filename: null }, { status: 500 })
  }
}
