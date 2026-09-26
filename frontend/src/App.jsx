import { useEffect, useMemo, useRef, useState } from 'react'
import clarityFiLogo from './assets/clarityfi-logo-parrot-green-light.png'
import {
  AudioLines,
  ArrowLeft,
  ArrowRight,
  BadgeDollarSign,
  BarChart3,
  BookOpen,
  Bot,
  BriefcaseBusiness,
  Calculator,
  Info,
  Check,
  CircleDollarSign,
  Compass,
  FileText,
  Film,
  Flag,
  HelpCircle,
  Landmark,
  Lightbulb,
  LineChart,
  LibraryBig,
  LockKeyhole,
  LogOut,
  Mail,
  Mic,
  MicOff,
  Newspaper,
  PieChart,
  Plus,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Target,
  UserRound,
  Volume2,
  VolumeX,
  WalletCards,
} from 'lucide-react'

const tabs = [
  { id: 'qa', label: 'Finance Q&A', icon: HelpCircle },
  { id: 'portfolio', label: 'Portfolio Analysis', icon: PieChart },
  { id: 'market', label: 'Market Analysis', icon: LineChart },
  { id: 'goals', label: 'Goal Planning', icon: Target },
  { id: 'news', label: 'News Synthesizer', icon: Newspaper },
  { id: 'tax', label: 'Tax Education', icon: FileText },
  { id: 'glossary', label: 'Glossary', icon: LibraryBig },
]

function LogoMenu({ compact = false }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const onDocClick = (event) => { if (ref.current && !ref.current.contains(event.target)) setOpen(false) }
    const onKey = (event) => { if (event.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onDocClick); document.removeEventListener('keydown', onKey) }
  }, [open])

  return (
    <div className={`brand-lockup logo-menu${compact ? ' compact' : ''}`} ref={ref}>
      <button type="button" className="logo-menu-trigger" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <img className="brand-logo-image" src={clarityFiLogo} alt="ClarityFi" />
      </button>
      {open && (
        <div className="logo-menu-dropdown" role="menu">
          <a role="menuitem" href="/docs/design.html" target="_blank" rel="noopener noreferrer" onClick={() => setOpen(false)}>
            <FileText size={15} /> Design Document
          </a>
          <a role="menuitem" href="/presentation/ClarityFi_Capstone_Presentation.mp4" target="_blank" rel="noopener noreferrer" onClick={() => setOpen(false)}>
            <Film size={15} /> Recorded Demo
          </a>
        </div>
      )}
    </div>
  )
}

const formatCurrency = (value, digits = 0) =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: digits,
  }).format(Number(value) || 0)

function Pill({ children, tone = 'green' }) {
  return <span className={`pill pill-${tone}`}>{children}</span>
}

function EmptyState({ icon: Icon, title, body, action, onAction }) {
  return (
    <div className="empty-state">
      <div className="empty-icon"><Icon size={24} /></div>
      <h3>{title}</h3>
      <p>{body}</p>
      {action && <button className="button secondary" onClick={onAction}>{action}</button>}
    </div>
  )
}

function UserGate({ onSignedIn, notice = '' }) {
  const [mode, setMode] = useState('signin')
  const [userId, setUserId] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [email, setEmail] = useState('')
  const [resetToken, setResetToken] = useState('')
  const [forgotSent, setForgotSent] = useState(false)
  const [resetDone, setResetDone] = useState(false)
  const [newPassword, setNewPassword] = useState('')
  const [confirmNewPassword, setConfirmNewPassword] = useState('')
  const [touched, setTouched] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [config, setConfig] = useState(null)
  const signup = mode === 'signup'
  const minLength = config?.min_password_length || 8

  useEffect(() => {
    apiFetch('/api/auth/config').then((r) => (r.ok ? r.json() : null)).then((data) => data && setConfig(data)).catch(() => {})
  }, [])

  // A password-reset email links here as .../?reset_token=... — pick it up once, then scrub it from the
  // address bar so it doesn't linger in browser history longer than needed.
  useEffect(() => {
    const token = new URLSearchParams(window.location.search).get('reset_token')
    if (!token) return
    setResetToken(token)
    setMode('reset')
    const url = new URL(window.location.href)
    url.searchParams.delete('reset_token')
    window.history.replaceState({}, '', url.toString())
  }, [])

  const idOk = userId.trim().length >= 3
  const passwordOk = signup ? password.length >= minLength : password.length > 0
  const confirmOk = !signup || password === confirm
  const valid = idOk && passwordOk && confirmOk

  const submit = async (event) => {
    event.preventDefault()
    setTouched(true)
    setError('')
    if (!valid) return
    setBusy(true)
    try {
      const body = { user_id: userId.trim(), password }
      if (signup && email.trim()) body.email = email.trim()
      const response = await apiFetch(signup ? '/api/auth/register' : '/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await response.json().catch(() => null)
      if (!response.ok) throw new Error(typeof data?.detail === 'string' ? data.detail : 'Could not sign in. Please try again.')
      setPassword('')
      setConfirm('')
      setEmail('')
      onSignedIn(data)
    } catch (err) {
      setError(err.message === 'Failed to fetch' ? 'Could not reach the ClarityFI service. Is the backend running?' : err.message)
    } finally {
      setBusy(false)
    }
  }

  const emailOk = /\S+@\S+\.\S+/.test(email.trim())

  const submitForgot = async (event) => {
    event.preventDefault()
    setTouched(true)
    setError('')
    if (!emailOk) return
    setBusy(true)
    try {
      const response = await apiFetch('/api/auth/forgot-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim() }),
      })
      if (response.status === 429) {
        const data = await response.json().catch(() => null)
        throw new Error(typeof data?.detail === 'string' ? data.detail : 'Too many requests. Please wait and try again.')
      }
      setForgotSent(true)
    } catch (err) {
      setError(err.message === 'Failed to fetch' ? 'Could not reach the ClarityFI service. Is the backend running?' : err.message)
    } finally {
      setBusy(false)
    }
  }

  const newPasswordOk = newPassword.length >= minLength
  const confirmNewOk = newPassword === confirmNewPassword

  const submitReset = async (event) => {
    event.preventDefault()
    setTouched(true)
    setError('')
    if (!newPasswordOk || !confirmNewOk) return
    setBusy(true)
    try {
      const response = await apiFetch('/api/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: resetToken, new_password: newPassword }),
      })
      if (!response.ok) {
        const data = await response.json().catch(() => null)
        throw new Error(typeof data?.detail === 'string' ? data.detail : 'This reset link is invalid or has expired.')
      }
      setNewPassword('')
      setConfirmNewPassword('')
      setResetDone(true)
    } catch (err) {
      setError(err.message === 'Failed to fetch' ? 'Could not reach the ClarityFI service. Is the backend running?' : err.message)
    } finally {
      setBusy(false)
    }
  }

  const goToSignIn = () => {
    setMode('signin')
    setError('')
    setTouched(false)
    setForgotSent(false)
    setResetDone(false)
    setEmail('')
    setResetToken('')
  }

  const useDemo = (id) => {
    setMode('signin')
    setUserId(id)
    setPassword(config?.demo_password || '')
    setTouched(false)
    setError('')
  }

  return (
    <main className="gate-page">
      <section className="gate-shell">
        <LogoMenu />

        <div className="gate-orbit-row" aria-hidden="true">
          <div className="gate-orbit orbit-one"><CircleDollarSign size={28} /></div>
          <div className="gate-orbit orbit-two"><Compass size={24} /></div>
          <div className="gate-orbit orbit-three"><BookOpen size={24} /></div>
        </div>

        <div className="gate-copy">
          <Pill tone="purple"><Sparkles size={13} /> Guided financial learning</Pill>
          <h1>Understand money.<br />Plan with confidence.</h1>
          <p>Explore financial concepts, test a sample portfolio, and turn goals into practical next steps—all in one friendly workspace.</p>
        </div>

        {mode === 'forgot' && (
          <form className="user-card" onSubmit={submitForgot} noValidate>
            <div className="step-label"><span>1</span> Reset your password</div>
            {forgotSent ? (
              <>
                <div className="gate-notice" role="status">
                  If that email is on an account, we've sent a link to reset your password. It's valid for a little
                  while, so check your inbox soon.
                </div>
                <button className="button primary full" type="button" onClick={goToSignIn}>Back to sign in <ArrowRight size={18} /></button>
              </>
            ) : (
              <>
                <label htmlFor="forgot-email">Email</label>
                <div className={`input-wrap ${touched && !emailOk ? 'error' : ''}`}>
                  <Mail size={19} />
                  <input id="forgot-email" type="email" value={email} onChange={(event) => setEmail(event.target.value)}
                    placeholder="you@example.com" autoComplete="email" autoFocus />
                </div>
                {touched && !emailOk && <span className="field-error">Enter a valid email address.</span>}
                {error && <div className="field-error form-error" role="alert">{error}</div>}
                <button className="button primary full" type="submit" disabled={busy}>
                  {busy ? 'Please wait…' : 'Send reset link'} <ArrowRight size={18} />
                </button>
                <button className="text-button" type="button" onClick={goToSignIn}><ArrowLeft size={15} /> Back to sign in</button>
              </>
            )}
          </form>
        )}

        {mode === 'reset' && (
          <form className="user-card" onSubmit={submitReset} noValidate>
            <div className="step-label"><span>1</span> Choose a new password</div>
            {resetDone ? (
              <>
                <div className="gate-notice" role="status">Your password has been updated. Please sign in again.</div>
                <button className="button primary full" type="button" onClick={goToSignIn}>Go to sign in <ArrowRight size={18} /></button>
              </>
            ) : (
              <>
                <label htmlFor="new-password">New password</label>
                <div className={`input-wrap ${touched && !newPasswordOk ? 'error' : ''}`}>
                  <LockKeyhole size={19} />
                  <input id="new-password" type="password" value={newPassword} onChange={(event) => setNewPassword(event.target.value)}
                    placeholder={`At least ${minLength} characters`} autoComplete="new-password" autoFocus />
                </div>
                {touched && !newPasswordOk && <span className="field-error">Use at least {minLength} characters.</span>}
                <label htmlFor="confirm-new-password">Confirm new password</label>
                <div className={`input-wrap ${touched && !confirmNewOk ? 'error' : ''}`}>
                  <LockKeyhole size={19} />
                  <input id="confirm-new-password" type="password" value={confirmNewPassword}
                    onChange={(event) => setConfirmNewPassword(event.target.value)} autoComplete="new-password" />
                </div>
                {touched && !confirmNewOk && <span className="field-error">The passwords do not match.</span>}
                {error && <div className="field-error form-error" role="alert">{error}</div>}
                <button className="button primary full" type="submit" disabled={busy}>
                  {busy ? 'Please wait…' : 'Update password'} <ArrowRight size={18} />
                </button>
                <button className="text-button" type="button" onClick={goToSignIn}><ArrowLeft size={15} /> Back to sign in</button>
              </>
            )}
          </form>
        )}

        {(mode === 'signin' || mode === 'signup') && (
          <form className="user-card" onSubmit={submit} noValidate>
            <div className="step-label"><span>1</span> {signup ? 'Create your account' : 'Ready to explore?'}</div>
            {notice && <div className="gate-notice" role="status">{notice}</div>}
            <label htmlFor="user-id">User ID</label>
            <div className={`input-wrap ${touched && !idOk ? 'error' : ''}`}>
              <UserRound size={19} />
              <input id="user-id" value={userId} onChange={(event) => setUserId(event.target.value)} onBlur={() => setTouched(true)}
                placeholder="e.g. learner-204" autoComplete="username" autoFocus />
            </div>
            {touched && !idOk && <span className="field-error">Enter at least 3 characters.</span>}
            {signup && (
              <>
                <label htmlFor="signup-email">Email (optional, for password reset)</label>
                <div className="input-wrap">
                  <Mail size={19} />
                  <input id="signup-email" type="email" value={email} onChange={(event) => setEmail(event.target.value)}
                    placeholder="you@example.com" autoComplete="email" />
                </div>
              </>
            )}
            <label htmlFor="password">Password</label>
            <div className={`input-wrap ${touched && !passwordOk ? 'error' : ''}`}>
              <LockKeyhole size={19} />
              <input id="password" type="password" value={password} onChange={(event) => setPassword(event.target.value)}
                placeholder={signup ? `At least ${minLength} characters` : 'Your password'} autoComplete={signup ? 'new-password' : 'current-password'} />
            </div>
            {touched && !passwordOk && <span className="field-error">{signup ? `Use at least ${minLength} characters.` : 'Enter your password.'}</span>}
            {signup && (
              <>
                <label htmlFor="confirm-password">Confirm password</label>
                <div className={`input-wrap ${touched && !confirmOk ? 'error' : ''}`}>
                  <LockKeyhole size={19} />
                  <input id="confirm-password" type="password" value={confirm} onChange={(event) => setConfirm(event.target.value)} autoComplete="new-password" />
                </div>
                {touched && !confirmOk && <span className="field-error">The passwords do not match.</span>}
              </>
            )}
            {!signup && (
              <button className="text-button" type="button" onClick={() => { setMode('forgot'); setError(''); setTouched(false) }}>
                Forgot password?
              </button>
            )}
            {error && <div className="field-error form-error" role="alert">{error}</div>}
            <button className="button primary full" type="submit" disabled={busy}>
              {busy ? 'Please wait…' : signup ? 'Create account' : 'Enter ClarityFi'} <ArrowRight size={18} />
            </button>
            <button className="text-button" type="button" onClick={() => { setMode(signup ? 'signin' : 'signup'); setError(''); setTouched(false) }}>
              {signup ? 'Already have an account? Sign in' : 'New here? Create an account'}
            </button>
            {config?.demo_login && !signup && (
              <div className="demo-box">
                <small>Try a demo account (sample portfolio). Password: <b>{config.demo_password}</b></small>
                <div className="demo-chips">
                  {config.demo_users.slice(0, 5).map((id) => <button type="button" key={id} onClick={() => useDemo(id)}>{id}</button>)}
                </div>
              </div>
            )}
            <div className="privacy-note"><ShieldCheck size={15} /> Questions and uploaded content may be sent to OpenAI. Avoid sensitive information.</div>
          </form>
        )}
      </section>
    </main>
  )
}

const starterMessages = [
  {
    role: 'assistant',
    text: "Hi! I’m your learning guide. Ask a money question in plain English, and I’ll explain the idea without the jargon.",
  },
]

const navigationCommands = new Map([
  ['portfolio', 'portfolio'],
  ['portfolio analysis', 'portfolio'],
  ['portfolio analytics', 'portfolio'],
  ['porffolio analysis', 'portfolio'],
  ['porffolio analytics', 'portfolio'],
  ['porfolio analysis', 'portfolio'],
  ['porfolio analytics', 'portfolio'],
  ['market', 'market'],
  ['market analysis', 'market'],
  ['market analytics', 'market'],
  ['goal', 'goals'],
  ['goals', 'goals'],
  ['goal planning', 'goals'],
  ['news', 'news'],
  ['news synthesizer', 'news'],
  ['glossary', 'glossary'],
  ['financial glossary', 'glossary'],
  ['tax', 'tax'],
  ['tax education', 'tax'],
])

function navigationTarget(value) {
  let command = value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

  command = command
    .replace(/\s+please$/, '')
    .replace(/^please\s+/, '')
    .replace(/^(?:can|could|would)\s+you\s+/, '')
    .replace(/^please\s+/, '')
    .replace(/^i\s+(?:would|d)\s+like\s+to\s+/, '')
    .replace(/^i\s+want\s+to\s+/, '')
    .replace(/^(?:go|open|show|switch|navigate|take)\s+(?:me\s+)?(?:to\s+)?/, '')
    .replace(/^the\s+/, '')
    .replace(/\s+(?:tab|section|page)$/, '')
    .trim()

  return navigationCommands.get(command) ?? null
}

function answerQuestion(question) {
  const lower = question.toLowerCase()
  if (lower.includes('etf') || lower.includes('fund')) {
    return 'An ETF is a basket of investments that trades like a stock. One broad-market ETF can hold hundreds or thousands of companies, which makes diversification easier. The tradeoff: you still experience market ups and downs, and every fund has fees and a specific strategy to understand.'
  }
  if (lower.includes('bond')) {
    return 'A bond is essentially an IOU. You lend money to a government or company; in return, it generally pays interest and returns principal at maturity. Bond prices can still fall—especially when interest rates rise or the issuer’s credit quality weakens.'
  }
  if (lower.includes('risk') || lower.includes('divers')) {
    return 'Diversification spreads exposure across different investments so one setback has less impact. It cannot prevent every loss, but mixing asset classes, sectors, and regions can reduce concentration risk. A useful first check is whether any single holding dominates your portfolio.'
  }
  if (lower.includes('emergency') || lower.includes('saving')) {
    return 'An emergency fund is cash reserved for unplanned essential expenses. A common learning benchmark is several months of core costs, but the right buffer depends on income stability, insurance, and household needs. Keep it accessible rather than chasing maximum returns.'
  }
  return 'A helpful way to unpack that is to separate the goal, time horizon, and risk. Start by naming when you may need the money, how much loss you could tolerate, and whether the choice is diversified. That turns a broad money question into decisions you can compare.'
}

function FinanceQA({ userId, onNavigate, isAdmin = false }) {
  const [messages, setMessages] = useState(starterMessages)
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const [backend, setBackend] = useState({ checked: false, online: false, ready: false, chunks: 0 })
  const [upload, setUpload] = useState({ state: 'idle', message: '' })
  const [listening, setListening] = useState(false)
  const [speechError, setSpeechError] = useState('')
  const [speaking, setSpeaking] = useState(false)
  const recognitionRef = useRef(null)
  const recorderRef = useRef(null)
  const playerRef = useRef(null)
  const [hdVoice, setHdVoice] = useState(false)   // OpenAI Whisper/TTS instead of the browser's built-in voice
  const [transcribing, setTranscribing] = useState(false)

  const toggleHdVoice = async (enabled) => {
    setSpeechError('')
    if (!enabled) { setHdVoice(false); return }
    try {
      const status = await audioStatus()
      if (!status.transcription && !status.speech) throw new Error('HD voice is not available on this server.')
      setHdVoice(true)
    } catch (error) {
      setHdVoice(false)
      setSpeechError(`${error.message} Using your browser's voice instead.`)
    }
  }

  const checkBackend = async () => {
    try {
      const response = await apiFetch('/api/health')
      if (!response.ok) throw new Error('Backend unavailable')
      const data = await response.json()
      setBackend({ checked: true, online: true, ready: data.openai_ready, chunks: data.knowledge_chunks })
    } catch {
      setBackend({ checked: true, online: false, ready: false, chunks: 0 })
    }
  }

  useEffect(() => {
    checkBackend()
    return () => {
      recognitionRef.current?.abort()
      recorderRef.current?.stop()
      playerRef.current?.pause()
      window.speechSynthesis?.cancel()
    }
  }, [])

  const speakAnswer = (text) => {
    const spoken = text.replace(/\[\d+\]/g, '').replace(/[*_#`]/g, '').slice(0, 1500)
    if (hdVoice) {
      playerRef.current?.pause()
      setSpeaking(true)
      playSpeech(spoken, () => setSpeaking(false))
        .then((player) => { playerRef.current = player })
        .catch((error) => {
          setSpeaking(false)
          setSpeechError(`${error.message} Using your browser's voice instead.`)
          speakWithBrowser(text)
        })
      return
    }
    speakWithBrowser(text)
  }

  const speakWithBrowser = (text) => {
    if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) {
      setSpeechError('Spoken answers are not available in this browser.')
      return
    }
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel()
    }
    const spokenText = text
      .replace(/\[\d+\]/g, '')
      .replace(/[*_#`]/g, '')
      .slice(0, 5000)
    const utterance = new window.SpeechSynthesisUtterance(spokenText)
    utterance.lang = 'en-US'
    utterance.rate = 0.96
    utterance.onstart = () => setSpeaking(true)
    utterance.onend = () => setSpeaking(false)
    utterance.onerror = () => {
      setSpeaking(false)
      setSpeechError('The browser could not play this spoken answer.')
    }
    window.speechSynthesis.speak(utterance)
  }

  const stopSpeaking = () => {
    playerRef.current?.pause()
    window.speechSynthesis?.cancel()
    setSpeaking(false)
  }

  const ask = async (question, inputMode = 'text') => {
    const clean = question.trim()
    if (!clean) return

    const destination = navigationTarget(clean)
    if (destination) {
      setInput('')
      setSpeechError('')
      onNavigate(destination)
      return
    }

    if (thinking) return

    setMessages((current) => [...current, { role: 'user', text: clean, inputMode }])
    setInput('')
    setSpeechError('')
    setThinking(true)

    if (backend.online) {
      try {
        const response = await apiFetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            conversation_id: 'finance-qa',
            question: clean,
            top_k: 4,
            input_mode: inputMode,
            response_mode: inputMode === 'voice' ? 'voice' : 'text',
          }),
        })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || 'The OpenAI RAG service could not answer.')
        setMessages((current) => [...current, {
          role: 'assistant',
          text: data.answer,
          sources: data.sources,
          answerMode: data.response_mode,
          notice: data.notice || null,
          cacheHit: data.cache_hit || null,
        }])
        if (inputMode === 'voice') speakAnswer(data.notice ? `${data.answer} ${data.notice}` : data.answer)
      } catch (error) {
        setMessages((current) => [...current, {
          role: 'assistant',
          text: `I couldn’t query the finance knowledge base: ${error.message}`,
          error: true,
        }])
      } finally {
        setThinking(false)
      }
      return
    }

    window.setTimeout(() => {
      const answer = answerQuestion(clean)
      setMessages((current) => [...current, {
        role: 'assistant',
        text: answer,
        sample: true,
        answerMode: inputMode === 'voice' ? 'voice' : 'text',
      }])
      if (inputMode === 'voice') speakAnswer(answer)
      setThinking(false)
    }, 500)
  }

  const toggleRecording = async () => {
    if (listening) {
      recorderRef.current?.stop()
      return
    }
    setSpeechError('')
    try {
      recorderRef.current = await startRecording()
    } catch {
      setSpeechError('Microphone permission was blocked. Allow it in your browser settings and try again.')
      return
    }
    setListening(true)
    const recorder = recorderRef.current
    const blob = await recorder.finished
    recorderRef.current = null
    setListening(false)
    setTranscribing(true)
    try {
      const text = await transcribe(blob)
      if (text) ask(text, 'voice')
      else setSpeechError("I couldn't hear a question. Please try again.")
    } catch (error) {
      setSpeechError(`${error.message} You can type your question or turn off HD voice.`)
    } finally {
      setTranscribing(false)
    }
  }

  const toggleListening = () => {
    if (hdVoice && canRecord()) {
      toggleRecording()
      return
    }
    if (listening) {
      recognitionRef.current?.stop()
      return
    }

    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!Recognition) {
      setSpeechError('Voice input is not supported here. Try the latest Chrome or Safari on your Mac.')
      return
    }

    setSpeechError('')
    let finalTranscript = ''
    const recognition = new Recognition()
    recognition.lang = 'en-US'
    recognition.continuous = false
    recognition.interimResults = true
    recognition.maxAlternatives = 1
    recognition.onstart = () => setListening(true)
    recognition.onresult = (event) => {
      let currentTranscript = ''
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const text = event.results[index][0].transcript
        currentTranscript += text
        if (event.results[index].isFinal) finalTranscript += text
      }
      setInput((finalTranscript || currentTranscript).trim())
    }
    recognition.onerror = (event) => {
      const message = event.error === 'not-allowed'
        ? 'Microphone permission was blocked. Allow it in your browser settings and try again.'
        : `Voice input stopped: ${event.error || 'unknown error'}.`
      setSpeechError(message)
      setListening(false)
    }
    recognition.onend = () => {
      setListening(false)
      recognitionRef.current = null
      if (finalTranscript.trim()) ask(finalTranscript, 'voice')
    }
    recognitionRef.current = recognition
    recognition.start()
  }

  const ingestFile = async (event) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setUpload({ state: 'loading', message: `Loading ${file.name}…` })
    try {
      const body = new FormData()
      body.append('file', file)
      body.append('reset', 'false')
      const response = await apiFetch('/api/ingest/file', { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Ingestion failed')
      setUpload({ state: 'success', message: `${data.documents} documents · ${data.chunks_ingested} chunks added` })
      setBackend((current) => ({ ...current, chunks: data.total_chunks }))
      setMessages((current) => [...current, {
        role: 'assistant',
        text: `${file.name} is ready. I added ${data.chunks_ingested} searchable chunks to your local FAISS knowledge base.`,
      }])
    } catch (error) {
      setUpload({ state: 'error', message: error.message })
    }
  }

  return (
    <div className="feature-grid qa-grid">
      <section className="panel chat-panel">
        <div className="panel-heading">
          <div>
            <Pill tone={backend.online ? 'green' : 'purple'}>
              <Bot size={13} /> {backend.online ? 'OpenAI RAG connected' : 'Sample mode'}
            </Pill>
            <h2>Ask a finance question</h2>
            <p>Type or use the microphone; voice questions receive spoken answers.</p>
          </div>
        </div>

        <div className="messages" aria-live="polite">
          {messages.map((message, index) => (
            <div key={`${message.role}-${index}`} className={`message-row ${message.role}`}>
              {message.role === 'assistant' && <div className="avatar"><Sparkles size={15} /></div>}
              <div className={`message ${message.error ? 'message-error' : ''}`}>
                {message.inputMode === 'voice' && <small className="message-mode"><Mic size={11} /> Voice question</small>}
                {message.text}
                {message.notice && (
                  <div className={`data-notice ${message.cacheHit ? 'cached' : ''}`} role="note">
                    <Info size={13} aria-hidden="true" /> <span>{message.notice}</span>
                  </div>
                )}
                {message.sources?.length > 0 && (
                  <div className="message-sources">
                    <strong>Sources</strong>
                    {message.sources.map((source) => (
                      <span key={`${source.index}-${source.title}`}>
                        [{source.index}]{' '}
                        {source.url
                          ? <a href={source.url} target="_blank" rel="noopener noreferrer">{source.title}</a>
                          : source.title}
                        {source.source ? ` · ${source.source}` : ''}{source.category ? ` · ${source.category}` : ''}
                      </span>
                    ))}
                  </div>
                )}
                {message.sample && <small className="sample-label">Sample response · start the backend for RAG</small>}
                {message.role === 'assistant' && !message.error && (
                  <button className="listen-button" onClick={() => speakAnswer(message.notice ? `${message.text} ${message.notice}` : message.text)}>
                    <Volume2 size={13} /> Listen
                  </button>
                )}
              </div>
            </div>
          ))}
          {thinking && (
            <div className="message-row assistant">
              <div className="avatar"><Sparkles size={15} /></div>
              <div className="message typing"><i /><i /><i /></div>
            </div>
          )}
        </div>

        <div className="prompt-chips">
          {['How do ETFs work?', 'Why diversify?', 'What are bonds?'].map((prompt) => (
            <button key={prompt} onClick={() => ask(prompt)}>{prompt}</button>
          ))}
        </div>

        {listening && (
          <div className="voice-status" role="status">
            <AudioLines size={17} />
            <strong>Listening…</strong>
            <span className="voice-bars"><i /><i /><i /><i /></span>
            <small>{hdVoice && canRecord() ? 'Speak, then press the mic again to send.' : 'Speak naturally, then pause.'}</small>
          </div>
        )}
        {transcribing && <div className="voice-status" role="status"><AudioLines size={17} /><strong>Transcribing…</strong></div>}
        {speechError && <div className="speech-error" role="alert">{speechError}</div>}

        <form className="composer" onSubmit={(event) => { event.preventDefault(); ask(input, 'text') }}>
          <input
            aria-label="Ask a finance question"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about ETFs, bonds, saving…"
          />
          <button
            className={`icon-button voice-button ${listening ? 'listening' : ''}`}
            title={listening ? 'Stop listening' : 'Ask by voice'}
            aria-label={listening ? 'Stop listening' : 'Ask by voice'}
            aria-pressed={listening}
            type="button"
            onClick={toggleListening}
          >
            {listening ? <MicOff size={18} /> : <Mic size={18} />}
          </button>
          <button className="icon-button send-button" title="Send question" aria-label="Send question" type="submit">
            <Send size={18} />
          </button>
        </form>
        <div className="composer-note">
          <span><Mic size={12} /> Voice questions receive spoken answers automatically.</span>
          <label className="hd-toggle"><input type="checkbox" checked={hdVoice} onChange={(event) => toggleHdVoice(event.target.checked)} /> HD voice (OpenAI)</label>
          {speaking && <button onClick={stopSpeaking}><VolumeX size={12} /> Stop speaking</button>}
        </div>
      </section>

      <aside className="side-stack">
        <section className="panel topic-card knowledge-card">
          <div className="section-kicker"><BookOpen size={16} /> Local FAISS knowledge base</div>
          <div className={`connection-row ${backend.online ? 'connected' : ''}`}>
            <span />
            <strong>{backend.checked ? (backend.online ? 'FastAPI connected' : 'Backend offline') : 'Checking backend…'}</strong>
          </div>
          <p>{backend.online ? `${backend.chunks} searchable chunks in FAISS` : 'Start the local launcher to enable RAG and JSON ingestion.'}</p>
          {backend.online && !backend.ready && <p className="mini-warning">OpenAI is unavailable. Check your API key and internet connection.</p>}
          {isAdmin ? (
            <label className={`button secondary full upload-button ${!backend.online ? 'disabled' : ''}`}>
              <Plus size={16} /> Add JSON knowledge
              <input type="file" accept=".json,.jsonl,.ndjson,application/json" onChange={ingestFile} disabled={!backend.online} />
            </label>
          ) : <p className="callout-copy">Adding knowledge files is limited to administrators.</p>}
          {upload.message && <div className={`upload-status ${upload.state}`}>{upload.state === 'success' && <Check size={14} />}{upload.message}</div>}
          {!backend.online && <button className="text-button" onClick={checkBackend}>Check connection</button>}
        </section>
        <section className="tip-card">
          <Lightbulb size={20} />
          <div><strong>Learning tip</strong><p>Ask “what could go wrong?” to surface an investment’s less obvious risks.</p></div>
        </section>
      </aside>
    </div>
  )
}

function Glossary() {
  const [state, setState] = useState({ loading: true, online: true, initialized: false, count: 0, items: [], error: '' })
  const [query, setQuery] = useState('')
  const [letter, setLetter] = useState('All')
  const [upload, setUpload] = useState({ state: 'idle', message: '' })

  const loadGlossary = async () => {
    setState((current) => ({ ...current, loading: true, error: '' }))
    try {
      const response = await apiFetch('/api/glossary')
      if (!response.ok) throw new Error('Glossary service is unavailable')
      const data = await response.json()
      setState({
        loading: false,
        online: true,
        initialized: data.initialized,
        count: data.count,
        items: data.items,
        error: data.initialization_error || '',
      })
    } catch {
      setState({ loading: false, online: false, initialized: false, count: 0, items: [], error: 'Start the local FastAPI service to load the glossary.' })
    }
  }

  useEffect(() => {
    loadGlossary()
  }, [])

  const initials = useMemo(() => {
    const values = new Set(state.items.map((item) => item.term.charAt(0).toUpperCase()).filter(Boolean))
    return ['All', ...Array.from(values).sort()]
  }, [state.items])

  const visibleItems = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return state.items.filter((item) => {
      const matchesQuery = !needle || item.term.toLowerCase().includes(needle) || item.definition.toLowerCase().includes(needle)
      const matchesLetter = letter === 'All' || item.term.toUpperCase().startsWith(letter)
      return matchesQuery && matchesLetter
    })
  }, [state.items, query, letter])

  const initializeGlossary = async (event) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setUpload({ state: 'loading', message: `Loading ${file.name} once…` })
    try {
      const body = new FormData()
      body.append('file', file)
      const response = await apiFetch('/api/glossary/initialize', { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Glossary initialization failed')
      setUpload({ state: 'success', message: `${data.count} glossary terms loaded and locked for reference.` })
      await loadGlossary()
    } catch (error) {
      setUpload({ state: 'error', message: error.message })
    }
  }

  return (
    <section className="panel glossary-panel">
      <div className="panel-heading split glossary-heading">
        <div>
          <Pill tone="purple"><LibraryBig size={13} /> Reference library</Pill>
          <h2>Financial glossary</h2>
          <p>Plain-language terms loaded once from financial_glossary.json.</p>
        </div>
        {state.initialized && (
          <div className="locked-status"><LockKeyhole size={16} /><div><strong>Read-only reference</strong><span>{state.count} terms</span></div></div>
        )}
      </div>

      {!state.online ? (
        <EmptyState icon={LibraryBig} title="Glossary service is offline" body={state.error} action="Check again" onAction={loadGlossary} />
      ) : !state.initialized && !state.loading ? (
        <div className="glossary-initialize">
          <div className="empty-icon"><LibraryBig size={24} /></div>
          <h3>Load your glossary once</h3>
          <p>Select <strong>financial_glossary.json</strong>. Only term and definition are stored; the reference becomes read-only after import.</p>
          <label className="button primary glossary-upload">
            <Plus size={17} /> Choose glossary JSON
            <input type="file" accept=".json,.jsonl,.ndjson,application/json" onChange={initializeGlossary} />
          </label>
          {state.error && <div className="speech-error">{state.error}</div>}
          {upload.message && <div className={`upload-status ${upload.state}`}>{upload.state === 'success' && <Check size={14} />}{upload.message}</div>}
        </div>
      ) : (
        <>
          <div className="glossary-tools">
            <label className="glossary-search">
              <Search size={17} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search a term or definition" aria-label="Search glossary" />
            </label>
            <span>{visibleItems.length} result{visibleItems.length === 1 ? '' : 's'}</span>
          </div>

          <div className="alphabet-filter" aria-label="Filter glossary by first letter">
            {initials.map((initial) => (
              <button key={initial} className={letter === initial ? 'active' : ''} onClick={() => setLetter(initial)}>{initial}</button>
            ))}
          </div>

          {state.loading ? (
            <div className="glossary-loading">Loading glossary…</div>
          ) : visibleItems.length ? (
            <div className="glossary-list">
              <div className="glossary-row glossary-columns"><strong>Term</strong><strong>Definition</strong></div>
              {visibleItems.map((item) => (
                <article className="glossary-row" key={item.term}>
                  <h3>{item.term}</h3>
                  <p>{item.definition}</p>
                </article>
              ))}
            </div>
          ) : (
            <EmptyState icon={Search} title="No matching terms" body="Try a different word or choose All." />
          )}
          {upload.message && <div className={`upload-status ${upload.state}`}>{upload.message}</div>}
        </>
      )}
    </section>
  )
}

import { audioStatus, canRecord, playSpeech, startRecording, transcribe } from './audio'
import { UNAUTHORIZED_EVENT, apiFetch } from './http'
import PortfolioAnalysis from './tabs/Portfolio'
import MarketAnalysis from './tabs/Market'
import GoalPlanning from './tabs/Goals'
import NewsSynthesizer from './tabs/News'
import TaxEducation from './tabs/Tax'

const sections = {
  qa: FinanceQA,
  portfolio: PortfolioAnalysis,
  market: MarketAnalysis,
  goals: GoalPlanning,
  news: NewsSynthesizer,
  glossary: Glossary,
  tax: TaxEducation,
}

function Workspace({ userId, isAdmin, onExit }) {
  const [activeTab, setActiveTab] = useState('qa')
  const [commandNavigation, setCommandNavigation] = useState(false)
  const [goalsRefreshKey, setGoalsRefreshKey] = useState(0)
  const ActiveSection = activeTab === 'qa' ? null : sections[activeTab]
  const activeLabel = tabs.find((tab) => tab.id === activeTab)?.label

  const selectTab = (tabId) => {
    setActiveTab(tabId)
    setCommandNavigation(false)
    if (tabId === 'goals') setGoalsRefreshKey((k) => k + 1)
  }

  const navigateFromFinanceQA = (tabId) => {
    setActiveTab(tabId)
    setCommandNavigation(true)
    if (tabId === 'goals') setGoalsRefreshKey((k) => k + 1)
  }

  const returnToFinanceQA = () => {
    setActiveTab('qa')
    setCommandNavigation(false)
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <LogoMenu compact />
        <div className="header-context"><span>Learning workspace</span><b>/</b><strong>{activeLabel}</strong></div>
        <div className="user-menu"><span className="user-avatar">{userId.slice(0, 2).toUpperCase()}</span><div><small>User ID</small><strong>{userId}</strong></div><button className="icon-button" title="Sign out" aria-label="Sign out" onClick={onExit}><LogOut size={17} /></button></div>
      </header>

      <nav className="tabs" aria-label="Financial tools">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button key={id} className={activeTab === id ? 'active' : ''} onClick={() => selectTab(id)} aria-current={activeTab === id ? 'page' : undefined}>
            <Icon size={18} /><span>{label}</span>
          </button>
        ))}
      </nav>

      <div className={`content-wrap ${activeTab === 'qa' ? 'qa-active' : 'standard-active'}`}>
        <div className="welcome-row">
          <div><span className="eyebrow">Hello, {userId}</span><h1>{activeLabel}</h1></div>
          <div className="education-label"><BookOpen size={16} /> Educational only · Not financial advice</div>
        </div>
        <div className="workspace-panels">
          <div className="workspace-tab-panel" hidden={activeTab !== 'qa'}>
            <FinanceQA userId={userId} isAdmin={isAdmin} onNavigate={navigateFromFinanceQA} />
          </div>
          {ActiveSection && (
            <div className="workspace-tab-panel">
              {commandNavigation && (
                <button className="button secondary command-return-button" type="button" onClick={returnToFinanceQA}>
                  <ArrowLeft size={17} /> Back to Finance Q&amp;A
                </button>
              )}
              <ActiveSection userId={userId} {...(activeTab === 'goals' ? { refreshKey: goalsRefreshKey } : {})} />
            </div>
          )}
        </div>
      </div>
    </main>
  )
}

export default function App() {
  const [session, setSession] = useState({ status: 'loading', user: null, notice: '' })

  // Restore an existing sign-in (the cookie survives reloads); a 401 from any data call ends the session.
  useEffect(() => {
    let alive = true
    apiFetch('/api/auth/me')
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => { if (alive) setSession({ status: data?.user_id ? 'in' : 'out', user: data?.user_id ? data : null, notice: '' }) })
      .catch(() => { if (alive) setSession({ status: 'out', user: null, notice: '' }) })
    const expired = () => setSession((s) => (s.status === 'in' ? { status: 'out', user: null, notice: 'Your session ended. Please sign in again.' } : s))
    window.addEventListener(UNAUTHORIZED_EVENT, expired)
    return () => { alive = false; window.removeEventListener(UNAUTHORIZED_EVENT, expired) }
  }, [])

  const signOut = async () => {
    try { await apiFetch('/api/auth/logout', { method: 'POST' }) } catch { /* the local state is cleared regardless */ }
    setSession({ status: 'out', user: null, notice: '' })
  }

  if (session.status === 'loading') return <main className="gate-page"><p className="loading-line" role="status">Loading…</p></main>
  if (session.status === 'out') return <UserGate notice={session.notice} onSignedIn={(user) => setSession({ status: 'in', user, notice: '' })} />
  return <Workspace userId={session.user.user_id} isAdmin={Boolean(session.user.is_admin)} onExit={signOut} />
}
