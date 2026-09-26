import { cleanup, fireEvent, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { signIn, stubBackend } from './test/helpers'

afterEach(() => { cleanup(); vi.unstubAllGlobals(); delete window.MediaRecorder })

const ok = (data) => ({ ok: true, status: 200, json: async () => data })

function setup({ status = { transcription: true, speech: true }, transcript = 'How do ETFs work?', speakFails = false, statusFails = false } = {}) {
  const calls = []
  stubBackend(async (url, options = {}) => {
    calls.push(String(url))
    if (url === '/api/health') return ok({ openai_ready: true, knowledge_chunks: 24 })
    if (url === '/api/audio/status') return statusFails ? { ok: false, status: 503, json: async () => ({ detail: 'down' }) } : ok(status)
    if (url === '/api/audio/transcribe') return ok({ text: transcript })
    if (url === '/api/audio/speak') return speakFails ? { ok: false, status: 503, json: async () => ({ detail: 'Speech services are unavailable.' }) } : { ok: true, status: 200, blob: async () => new Blob(['mp3']) }
    if (url === '/api/chat') return ok({ answer: 'An ETF is a basket of investments.', response_mode: 'voice', sources: [] })
    return { ok: false, status: 404, json: async () => ({}) }
  })
  const play = vi.fn(async () => {})
  vi.stubGlobal('Audio', class { constructor() { this.pause = vi.fn() } play() { return play() } })
  URL.createObjectURL = vi.fn(() => 'blob:x')
  URL.revokeObjectURL = vi.fn()
  const stop = vi.fn()
  class FakeRecorder {
    constructor() { this.state = 'inactive'; this.mimeType = 'audio/webm' }
    start() { this.state = 'recording' }
    stop() { this.state = 'inactive'; this.ondataavailable?.({ data: new Blob(['x']) }); stop(); this.onstop?.() }
  }
  window.MediaRecorder = FakeRecorder
  const speechSynthesis = { speaking: false, speak: vi.fn((u) => { u.onstart?.(); u.onend?.() }), cancel: vi.fn() }
  vi.stubGlobal('SpeechSynthesisUtterance', class { constructor(t) { this.text = t } })
  vi.stubGlobal('speechSynthesis', speechSynthesis)
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn(async () => ({ getTracks: () => [{ stop: vi.fn() }] })) } })
  return { calls, play, speechSynthesis }
}

async function enter() {
  await signIn()
  await screen.findByText(/openai rag connected/i)
}

describe('HD voice (Whisper / TTS)', () => {
  it('records, transcribes on the server, asks, and plays the server voice', async () => {
    const { calls, play, speechSynthesis } = setup()
    await enter()
    fireEvent.click(screen.getByLabelText(/hd voice/i))
    await vi.waitFor(() => expect(calls).toContain('/api/audio/status'))
    fireEvent.click(await screen.findByRole('button', { name: /ask by voice/i }))
    fireEvent.click(await screen.findByRole('button', { name: /stop listening/i }))
    await screen.findByText(/basket of investments/i)
    await vi.waitFor(() => expect(play).toHaveBeenCalledOnce())
    expect(calls).toContain('/api/audio/transcribe')
    expect(calls).toContain('/api/audio/speak')
    expect(speechSynthesis.speak).not.toHaveBeenCalled()
  })

  it('falls back to the browser voice, and says so, when server speech fails', async () => {
    const { speechSynthesis } = setup({ speakFails: true })
    await enter()
    fireEvent.click(screen.getByLabelText(/hd voice/i))
    fireEvent.click(await screen.findByRole('button', { name: /ask by voice/i }))
    fireEvent.click(await screen.findByRole('button', { name: /stop listening/i }))
    await screen.findByText(/basket of investments/i)
    expect(await screen.findByRole('alert')).toHaveTextContent(/browser's voice instead/i)
    await vi.waitFor(() => expect(speechSynthesis.speak).toHaveBeenCalled())
  })

  it('stays on the browser voice when the server reports HD voice unavailable', async () => {
    setup({ statusFails: true })
    await enter()
    fireEvent.click(screen.getByLabelText(/hd voice/i))
    expect(await screen.findByRole('alert')).toHaveTextContent(/browser's voice instead/i)
    expect(screen.getByLabelText(/hd voice/i)).not.toBeChecked()
  })

  it('asks the user to retry when nothing was heard', async () => {
    const { calls } = setup({ transcript: '' })
    await enter()
    fireEvent.click(screen.getByLabelText(/hd voice/i))
    fireEvent.click(await screen.findByRole('button', { name: /ask by voice/i }))
    fireEvent.click(await screen.findByRole('button', { name: /stop listening/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/couldn't hear/i)
    expect(calls).not.toContain('/api/chat')
  })
})
