import { NextRequest, NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join } from 'path'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ filename: string }> }
) {
  try {
    const { filename } = await params
    
    if (!filename || !filename.endsWith('.pdf')) {
      return NextResponse.json(
        { error: 'Invalid file format' },
        { status: 400 }
      )
    }

    // Get the file from backend
    const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'
    
    const response = await fetch(`${BACKEND_URL}/files/${filename}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/pdf',
      },
    })

    if (!response.ok) {
      return NextResponse.json(
        { error: 'File not found' },
        { status: 404 }
      )
    }

    const fileBuffer = await response.arrayBuffer()

    return new NextResponse(fileBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'application/pdf',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
        'ETag': `"${Date.now()}"`, // Unique ETag to force fresh loads
        'Content-Disposition': `inline; filename="${filename}"`,
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
      },
    })
  } catch (error) {
    console.error('Error serving PDF file:', error)
    return NextResponse.json(
      { error: 'Failed to serve file' },
      { status: 500 }
    )
  }
}
