import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { signIn, stubBackend } from './test/helpers'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  delete window.SpeechRecognition
  delete window.webkitSpeechRecognition
})

describe('financial learning workspace', () => {
  it('enters the workspace and navigates all six tools', async () => {
    stubBackend()
    await signIn()
    expect(screen.getByRole('heading', { name: 'Finance Q&A' })).toBeInTheDocument()
    expect(document.querySelector('.content-wrap')).toHaveClass('qa-active')

    for (const label of ['Portfolio Analysis', 'Market Analysis', 'Goal Planning', 'News Synthesizer', 'Tax Education', 'Glossary']) {
      fireEvent.click(screen.getByRole('button', { name: label }))
      expect(screen.getByRole('heading', { name: label })).toBeInTheDocument()
      expect(document.querySelector('.content-wrap')).toHaveClass('standard-active')
    }
  })

  it('answers a sample finance question', async () => {
    stubBackend()
    await signIn()
    fireEvent.click(screen.getByRole('button', { name: 'How do ETFs work?' }))
    expect(screen.getAllByText('How do ETFs work?')).toHaveLength(2)
    await waitFor(() => expect(screen.getByText(/basket of investments/i)).toBeInTheDocument())
  })

  it('returns from command navigation with the Finance Q&A conversation preserved', async () => {
    stubBackend()
    await signIn()

    fireEvent.click(screen.getByRole('button', { name: 'How do ETFs work?' }))
    await screen.findByText(/basket of investments/i)

    fireEvent.change(screen.getByLabelText(/ask a finance question/i), {
      target: { value: 'porffolio analytics' },
    })
    fireEvent.click(screen.getByRole('button', { name: /send question/i }))

    expect(screen.getByRole('heading', { name: 'Portfolio Analysis' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /back to finance q&a/i }))

    expect(screen.getByRole('heading', { name: 'Finance Q&A' })).toBeInTheDocument()
    expect(screen.getByText(/basket of investments/i)).toBeInTheDocument()
  })

  it('supports a return path for every Finance Q&A navigation command', async () => {
    stubBackend()
    await signIn()

    const commands = [
      ['Can you take me to Market Analysis?', 'Market Analysis'],
      ['Open the Goal Planning tab', 'Goal Planning'],
      ['Please show me News', 'News Synthesizer'],
      ['Switch to the Glossary section', 'Glossary'],
      ['I want to go to Tax Education', 'Tax Education'],
    ]

    for (const [command, heading] of commands) {
      fireEvent.change(screen.getByLabelText(/ask a finance question/i), { target: { value: command } })
      fireEvent.click(screen.getByRole('button', { name: /send question/i }))
      expect(screen.getByRole('heading', { name: heading })).toBeInTheDocument()
      fireEvent.click(screen.getByRole('button', { name: /back to finance q&a/i }))
      expect(screen.getByRole('heading', { name: 'Finance Q&A' })).toBeInTheDocument()
    }
  })

  it('allows tab navigation while a Finance Q&A answer is pending', async () => {
    stubBackend()
    await signIn()

    fireEvent.click(screen.getByRole('button', { name: 'How do ETFs work?' }))
    fireEvent.change(screen.getByLabelText(/ask a finance question/i), {
      target: { value: 'Market Analysis' },
    })
    fireEvent.click(screen.getByRole('button', { name: /send question/i }))

    expect(screen.getByRole('heading', { name: 'Market Analysis' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /back to finance q&a/i })).toBeInTheDocument()
  })

  it('reloads saved goals every time the Goal Planning tab is clicked, even while already on it', async () => {
    let goalsCalls = 0
    stubBackend(async (url) => {
      if (url.startsWith('/api/goals/')) { goalsCalls += 1; return { ok: true, status: 200, json: async () => [], headers: { get: () => null } } }
      return { ok: false, status: 404, json: async () => ({ detail: 'not stubbed', code: 'x' }), headers: { get: () => null } }
    })
    await signIn()

    fireEvent.click(screen.getByRole('button', { name: 'Goal Planning' }))
    await waitFor(() => expect(goalsCalls).toBe(1))

    fireEvent.click(screen.getByRole('button', { name: 'Portfolio Analysis' }))
    fireEvent.click(screen.getByRole('button', { name: 'Goal Planning' }))
    await waitFor(() => expect(goalsCalls).toBe(2))

    // clicking the tab while it's already active should also refresh the saved-goals list
    fireEvent.click(screen.getByRole('button', { name: 'Goal Planning' }))
    await waitFor(() => expect(goalsCalls).toBe(3))
  })

  it('sends a browser transcript as voice mode and speaks the answer', async () => {
    const responses = {
      '/api/health': { openai_ready: true, knowledge_chunks: 24 },
      '/api/chat': {
        answer: 'An ETF is a diversified basket of investments. [1]',
        response_mode: 'voice',
        sources: [{ index: 1, title: 'ETF basics', category: 'ETFs' }],
      },
    }
    const fetchMock = stubBackend(async (url) => ({ ok: true, status: 200, json: async () => responses[url] }))

    class FakeUtterance {
      constructor(text) { this.text = text }
    }
    const speak = vi.fn((utterance) => {
      utterance.onstart?.()
      utterance.onend?.()
    })
    vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance)
    vi.stubGlobal('speechSynthesis', { speaking: false, speak, cancel: vi.fn() })

    class FakeRecognition {
      start() {
        this.onstart?.()
        const result = { 0: { transcript: 'How do ETFs work?' }, isFinal: true }
        this.onresult?.({ resultIndex: 0, results: [result] })
        this.onend?.()
      }
      stop() { this.onend?.() }
      abort() {}
    }
    window.SpeechRecognition = FakeRecognition

    await signIn()
    await screen.findByText(/openai rag connected/i)

    fireEvent.click(screen.getByRole('button', { name: /ask by voice/i }))
    await screen.findByText(/diversified basket/i)

    const chatCall = fetchMock.mock.calls.find(([url]) => url === '/api/chat')
    const request = JSON.parse(chatCall[1].body)
    expect(request.input_mode).toBe('voice')
    expect(request.response_mode).toBe('voice')
    expect(speak).toHaveBeenCalledOnce()
  })

  it('shows a visible notice, and speaks it, when an answer comes from the cache', async () => {
    const responses = {
      '/api/health': { openai_ready: true, knowledge_chunks: 200 },
      '/api/chat': {
        answer: 'An ETF is a basket of investments.',
        response_mode: 'voice',
        cache_hit: 'semantic',
        data_status: 'cached',
        notice: 'Answered from a saved response to a similar question ("What is an ETF?") (about 3 hours old). It was not regenerated for your exact wording.',
        sources: [{ index: 1, title: 'What Is an ETF?', category: 'etfs_and_funds', source: 'Investopedia', url: 'https://www.investopedia.com/terms/e/etf.asp' }],
      },
    }
    stubBackend(async (url) => ({ ok: true, status: 200, json: async () => responses[url] }))
    const spoken = []
    vi.stubGlobal('SpeechSynthesisUtterance', class { constructor(text) { this.text = text } })
    vi.stubGlobal('speechSynthesis', { speaking: false, speak: (u) => { spoken.push(u.text); u.onstart?.(); u.onend?.() }, cancel: vi.fn() })
    window.SpeechRecognition = class {
      start() { this.onstart?.(); this.onresult?.({ resultIndex: 0, results: [{ 0: { transcript: 'what are etfs' }, isFinal: true }] }); this.onend?.() }
      stop() { this.onend?.() }
      abort() {}
    }
    await signIn()
    await screen.findByText(/openai rag connected/i)
    fireEvent.click(screen.getByRole('button', { name: /ask by voice/i }))

    const note = await screen.findByRole('note')
    expect(note).toHaveTextContent(/saved response to a similar question/i)
    expect(note).toHaveTextContent(/about 3 hours old/i)
    await waitFor(() => expect(spoken.some((t) => t.includes('saved response to a similar question'))).toBe(true))
    const link = screen.getByRole('link', { name: 'What Is an ETF?' })
    expect(link).toHaveAttribute('href', 'https://www.investopedia.com/terms/e/etf.asp')
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'))
  })

  it('shows no notice for a freshly generated answer', async () => {
    const responses = {
      '/api/health': { openai_ready: true, knowledge_chunks: 200 },
      '/api/chat': { answer: 'A bond is a loan.', response_mode: 'voice', cache_hit: null, notice: null, sources: [] },
    }
    stubBackend(async (url) => ({ ok: true, status: 200, json: async () => responses[url] }))
    vi.stubGlobal('SpeechSynthesisUtterance', class { constructor(text) { this.text = text } })
    vi.stubGlobal('speechSynthesis', { speaking: false, speak: (u) => { u.onstart?.(); u.onend?.() }, cancel: vi.fn() })
    window.SpeechRecognition = class {
      start() { this.onstart?.(); this.onresult?.({ resultIndex: 0, results: [{ 0: { transcript: 'what is a bond' }, isFinal: true }] }); this.onend?.() }
      stop() { this.onend?.() }
      abort() {}
    }
    await signIn()
    await screen.findByText(/openai rag connected/i)
    fireEvent.click(screen.getByRole('button', { name: /ask by voice/i }))
    await screen.findByText(/a bond is a loan/i)
    expect(screen.queryByRole('note')).toBeNull()
  })
})
