import { api } from './api'
import { apiFetch } from './http'

export const canRecord = () => typeof navigator !== 'undefined' && Boolean(navigator.mediaDevices?.getUserMedia) && typeof window !== 'undefined' && Boolean(window.MediaRecorder)

export const audioStatus = () => api('/api/audio/status')

// Records from the microphone until stop() is called; resolves to a Blob. Nothing is stored anywhere.
export async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  const recorder = new window.MediaRecorder(stream)
  const chunks = []
  recorder.ondataavailable = (event) => { if (event.data?.size) chunks.push(event.data) }
  const finished = new Promise((resolve) => {
    recorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop())
      resolve(new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }))
    }
  })
  recorder.start()
  return { stop: () => { if (recorder.state !== 'inactive') recorder.stop() }, finished }
}

export async function transcribe(blob) {
  const body = new FormData()
  body.append('file', blob, 'speech.webm')
  const response = await apiFetch('/api/audio/transcribe', { method: 'POST', body })
  const data = await response.json().catch(() => null)
  if (!response.ok) throw new Error(data?.detail || 'Speech recognition is unavailable right now.')
  return (data?.text || '').trim()
}

// Returns an HTMLAudioElement that is already playing.
export async function playSpeech(text, onEnd) {
  const response = await apiFetch('/api/audio/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) })
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    throw new Error(data?.detail || 'The spoken answer is unavailable right now.')
  }
  const url = URL.createObjectURL(await response.blob())
  const player = new Audio(url)
  player.onended = () => { URL.revokeObjectURL(url); onEnd?.() }
  await player.play()
  return player
}
