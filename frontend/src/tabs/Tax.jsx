import { useState } from 'react'
import { BadgeDollarSign, ChevronDown, Landmark, Send, ShieldCheck } from 'lucide-react'
import { api, useAction } from '../api'
import { DataNotices, ErrorBox, Loading } from '../common'

const TOPICS = [
  { title: 'Marginal vs. effective rate', body: 'A marginal rate applies to the next portion of taxable income in a bracket. An effective rate compares total tax with a broader income amount. They answer different questions.' },
  { title: 'Deduction vs. credit', body: 'A deduction generally reduces income subject to tax. A credit generally reduces tax itself. Eligibility, limits, and refundability can change the result.' },
  { title: 'Why tax-advantaged accounts matter', body: 'Some accounts offer tax benefits now, later, or both. The rules depend on account type, eligibility, contribution limits, and withdrawal timing.' },
]
const EXAMPLES = ['What is the difference between a Roth and a traditional IRA?', 'How are capital gains taxed?', 'What is an HSA?']

export default function TaxEducation({ userId }) {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null)
  const [open, setOpen] = useState(0)
  const [action, run] = useAction()

  const ask = (text) => run(async () => {
    const q = (text ?? question).trim()
    if (!q) return
    setAnswer(await api('/api/chat', { method: 'POST', body: { user_id: userId, conversation_id: 'tax-education', question: q, top_k: 4 } }))
  })

  return (
    <div className="feature-grid tax-grid">
      <section className="panel tax-main">
        <div className="panel-heading"><span className="pill pill-yellow"><Landmark size={13} /> Tax concepts</span><h2>Learn the mechanics before the math</h2><p>Ask how a tax concept or account type works. Answers come from the knowledge base, with sources.</p></div>
        <form className="symbol-form" onSubmit={(e) => { e.preventDefault(); ask() }}>
          <input aria-label="Tax question" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. How does a Roth IRA work?" maxLength={500} />
          <button className="button primary" type="submit" disabled={action.pending}><Send size={15} /> Ask</button>
        </form>
        <div className="filter-row">{EXAMPLES.map((ex) => <button key={ex} type="button" onClick={() => { setQuestion(ex); ask(ex) }}>{ex}</button>)}</div>
        {action.pending && <Loading label="Looking that up…" />}
        <ErrorBox message={action.error} />
        {answer && (
          <div className="explanation-box">
            <div>
              <p className="pre-wrap">{answer.answer}</p>
              <DataNotices notice={answer.notice} />
              {answer.sources?.length > 0 && (
                <ul className="notes-list">{answer.sources.map((s, i) => (
                  <li key={i}>[{s.index ?? i + 1}] {/^https?:\/\//.test(s.url || '') ? <a href={s.url} target="_blank" rel="noreferrer noopener">{s.title}</a> : s.title}{s.source ? ` — ${s.source}` : ''}</li>
                ))}</ul>
              )}
            </div>
          </div>
        )}
        <div className="accordion-list">
          {TOPICS.map((topic, index) => (
            <button className={`accordion ${open === index ? 'open' : ''}`} key={topic.title} onClick={() => setOpen(open === index ? -1 : index)} aria-expanded={open === index}>
              <span>{topic.title}</span><ChevronDown size={17} />
              {open === index && <p>{topic.body}</p>}
            </button>
          ))}
        </div>
      </section>
      <aside className="side-stack">
        <section className="tip-card yellow"><BadgeDollarSign size={20} /><div><strong>Good question to ask</strong><p>“Which assumptions and tax year does this use?”</p></div></section>
        <section className="panel disclaimer-card">
          <div className="section-kicker"><ShieldCheck size={16} /> Educational guardrail</div>
          <p>This explains general concepts. It does not calculate your tax bill or give individualized tax advice.</p>
        </section>
      </aside>
    </div>
  )
}
