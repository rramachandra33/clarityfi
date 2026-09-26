import { useState } from 'react'
import { Download, Flag, Sparkles, Target, Trash2 } from 'lucide-react'
import { api, money, useAction, useApi } from '../api'
import { ErrorBox, LineChart } from '../common'

const num = (v) => (v === '' ? 0 : Number(v))

async function downloadSavedGoalsPdf(saved) {
  const { jsPDF } = await import('jspdf')
  const doc = new jsPDF()
  const marginX = 14
  let y = 18
  doc.setFontSize(16)
  doc.text('ClarityFI — Saved Goals', marginX, y)
  doc.setFontSize(10)
  doc.setTextColor(110)
  y += 7
  doc.text(`Generated ${new Date().toLocaleString()}`, marginX, y)
  y += 10
  doc.setTextColor(20)

  saved.forEach(({ goal }, index) => {
    if (y > 265) { doc.addPage(); y = 18 }
    doc.setFontSize(13)
    doc.text(`${index + 1}. ${goal.name}`, marginX, y)
    y += 7
    doc.setFontSize(11)
    const rows = [
      ['Target amount', money(goal.target_amount)],
      ['Target date', goal.target_date || '—'],
      ['Already saved', money(goal.current_savings || 0)],
      ['Monthly contribution', money(goal.monthly_contribution || 0)],
      ['Assumed return', `${goal.assumed_return_pct ?? 0}%`],
      ['Risk tolerance', goal.risk_tolerance || '—'],
    ]
    rows.forEach(([label, value]) => { doc.text(`${label}: ${value}`, marginX + 4, y); y += 6 })
    y += 4
  })

  doc.setFontSize(9)
  doc.setTextColor(120)
  doc.text('Educational only, not financial advice. Projections are illustrative, not predictions.', marginX, 285)
  doc.save('clarityfi-saved-goals.pdf')
}

export default function GoalPlanning({ userId, refreshKey = 0 }) {
  const id = encodeURIComponent(userId)
  const [form, setForm] = useState({ name: 'Home down payment', target_amount: 80000, current_savings: 24000, monthly_contribution: 900, years: 5, assumed_return_pct: 0 })
  const [result, setResult] = useState(null)
  const [nonce, setNonce] = useState(0)
  const [action, run] = useAction()
  const saved = useApi(`/api/goals/${id}`, `${refreshKey}:${nonce}`)
  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }))
  const body = (explain, f = form) => ({ name: f.name, target_amount: num(f.target_amount), current_savings: num(f.current_savings),
    monthly_contribution: num(f.monthly_contribution), years: num(f.years), assumed_return_pct: num(f.assumed_return_pct),
    user_id: userId, explain })

  const project = (explain) => run(async () => setResult(await api('/api/goals/project', { method: 'POST', body: body(explain) })))
  const save = () => run(async () => { await api(`/api/goals/${id}`, { method: 'POST', body: { ...body(false), explain: undefined, user_id: undefined } }); setNonce((n) => n + 1) })
  const remove = (goalId) => run(async () => { await api(`/api/goals/${id}/${goalId}`, { method: 'DELETE' }); setNonce((n) => n + 1) })
  const loadGoal = (goal) => {
    const yearsToGoal = (new Date(goal.target_date) - new Date()) / (365.25 * 24 * 3600 * 1000)
    const next = { name: goal.name, target_amount: goal.target_amount, current_savings: goal.current_savings ?? 0,
      monthly_contribution: goal.monthly_contribution ?? 0, years: Math.max(0.5, Math.round(yearsToGoal * 2) / 2), assumed_return_pct: goal.assumed_return_pct ?? 0 }
    setForm(next)
    run(async () => setResult(await api('/api/goals/project', { method: 'POST', body: body(false, next) })))
  }
  const expected = result?.projection.scenarios.find((s) => s.label === 'Your assumption') || result?.projection.scenarios[1]

  return (
    <div className="feature-grid goal-grid">
      <section className="panel goal-builder">
        <div className="panel-heading"><span className="pill pill-coral"><Flag size={13} /> Scenario builder</span><h2>Turn a goal into a monthly path</h2><p>Projections use assumed growth rates. They illustrate possibilities and are not predictions.</p></div>
        <div className="form-grid">
          <label>Goal name<input aria-label="Goal name" value={form.name} maxLength={80} onChange={set('name')} /></label>
          <label>Target amount<div className="money-input"><span>$</span><input aria-label="Target amount" type="number" min="1" step="1000" value={form.target_amount} onChange={set('target_amount')} /></div></label>
          <label>Already saved<div className="money-input"><span>$</span><input aria-label="Already saved" type="number" min="0" step="500" value={form.current_savings} onChange={set('current_savings')} /></div></label>
          <label>Monthly contribution<div className="money-input"><span>$</span><input aria-label="Monthly contribution" type="number" min="0" step="50" value={form.monthly_contribution} onChange={set('monthly_contribution')} /></div></label>
          <label>Years to goal<input aria-label="Years to goal" type="number" min="0.5" max="80" step="0.5" value={form.years} onChange={set('years')} /></label>
          <label>Assumed return (%)<input aria-label="Assumed return" type="number" min="-50" max="100" step="0.5" value={form.assumed_return_pct} onChange={set('assumed_return_pct')} /></label>
        </div>
        <p className="callout-copy">The table shows three scenarios centered on this number: your assumption, and +/- 2 percentage points for a below/above pair. 0% means no assumed growth.</p>
        <div className="button-row">
          <button className="button primary" disabled={action.pending} onClick={() => project(false)}>Project my goal</button>
          <button className="button secondary" disabled={action.pending} onClick={() => project(true)}><Sparkles size={15} /> Project and explain</button>
          <button className="button secondary" disabled={action.pending} onClick={save}>Save goal</button>
        </div>
        <ErrorBox message={action.error} />
        {result && (
          <div className="scenario-results">
            <table className="scenario-table">
              <thead><tr><th>Scenario</th><th>Assumed return</th><th>Projected value</th><th>Monthly needed</th><th>On track?</th></tr></thead>
              <tbody>{result.projection.scenarios.map((s) => (
                <tr key={s.label}><td>{s.label}</td><td>{(s.annual_return * 100).toFixed(1)}%</td><td>{money(s.projected_value)}</td><td>{money(s.required_monthly)}</td><td>{s.on_track ? 'Yes' : `Short by ${money(s.shortfall)}`}</td></tr>
              ))}</tbody>
            </table>
            <LineChart label="Projected savings by year for each scenario, with the target" format={(v) => money(v)}
              labels={(Object.values(result.projection.yearly_path)[0] || []).map((_, i) => (i === 0 ? 'Today' : `Year ${i}`))}
              refLine={{ value: result.goal.target_amount, label: `Target ${money(result.goal.target_amount)}` }}
              series={Object.entries(result.projection.yearly_path).map(([label, values]) => ({ label, values }))} />
            <ul className="notes-list">{result.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>
            {result.explanation && <div className="explanation-box"><Sparkles size={20} /><div><strong>In plain language</strong><p className="pre-wrap">{result.explanation}</p></div></div>}
            {result.explanation_error && <ErrorBox message={`Explanation unavailable: ${result.explanation_error}`} />}
          </div>
        )}
      </section>
      <aside className="panel result-card">
        <div className="result-icon"><Target size={25} /></div>
        <span>{expected ? 'Your assumption' : 'Your goal'}</span>
        <strong>{expected ? (expected.on_track ? 'On track' : 'Short of target') : '—'}</strong>
        <p>{expected ? `Projected ${money(expected.projected_value)} vs. target ${money(result.goal.target_amount)}.` : 'Run a projection to see the result.'}</p>
        <div className="rule" />
        <div className="panel-heading-row">
          <h3>Saved goals</h3>
          {saved.data?.length > 0 && (
            <button className="button secondary small" onClick={() => downloadSavedGoalsPdf(saved.data)}>
              <Download size={14} /> Download PDF
            </button>
          )}
        </div>
        {saved.data?.length === 0 && <small>No saved goals yet.</small>}
        {saved.data?.map(({ id: gid, goal }) => (
          <div className="saved-goal" key={gid}>
            <button type="button" className="saved-goal-info" disabled={action.pending} aria-label={`Load saved goal ${goal.name}`} onClick={() => loadGoal(goal)}>
              <strong>{goal.name}</strong><small>{money(goal.target_amount)} by {goal.target_date}</small>
            </button>
            <button className="icon-button" aria-label={`Delete ${goal.name}`} onClick={() => remove(gid)}><Trash2 size={15} /></button>
          </div>
        ))}
        <ErrorBox message={saved.error} />
      </aside>
    </div>
  )
}
