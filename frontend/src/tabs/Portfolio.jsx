import { useState } from 'react'
import { Check, ChevronDown, ShieldCheck, Sparkles } from 'lucide-react'
import { api, money, pct, useAction, useApi } from '../api'
import { DataNotices, ErrorBox, LineChart, Loading, StatusBadge } from '../common'

const TONES = ['teal', 'purple', 'yellow', 'coral', 'teal', 'purple']
const CLASS_COLORS = { stock: 'var(--green)', etf: 'var(--green)', bond: '#e4b04a', cash: 'var(--coral)', crypto: 'var(--purple)', fund: '#71c5a2' }
const RISKS = ['conservative', 'moderate', 'aggressive']
const BENCHMARKS = [['SPY', 'S&P 500 (SPY)'], ['QQQ', 'Nasdaq-100 (QQQ)'], ['VTI', 'Total US market (VTI)'], ['BND', 'US bonds (BND)']]
const shortDate = (iso) => new Date(`${iso}T00:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
const pts = (v) => (v === null || v === undefined ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)} pts`)

function donutBackground(byClass, highlight = null) {
  const entries = Object.entries(byClass || {}).filter(([, v]) => v > 0)
  let acc = 0
  const stops = entries.map(([name, v], i) => {
    const base = CLASS_COLORS[name] || ['var(--green)', 'var(--purple)', '#e4b04a', 'var(--coral)'][i % 4]
    const color = highlight && highlight !== name ? `color-mix(in srgb, ${base} 30%, white)` : base
    const from = acc
    acc += v
    return `${color} ${from}% ${Math.min(acc, 100)}%`
  })
  return stops.length ? `conic-gradient(${stops.join(', ')})` : 'var(--mint)'
}

function parseHoldings(text) {
  const holdings = []
  for (const line of text.split('\n').map((l) => l.trim()).filter(Boolean)) {
    const [ticker, shares] = line.split(/[\s,]+/)
    const n = Number(shares)
    if (!/^[A-Za-z0-9.\-^=]{1,12}$/.test(ticker || '') || !Number.isFinite(n) || n < 0) throw new Error(`Could not read "${line}". Use one "TICKER SHARES" per line.`)
    holdings.push({ ticker, shares: n })
  }
  return holdings
}

export default function PortfolioAnalysis({ userId }) {
  const id = encodeURIComponent(userId)
  const [nonce, setNonce] = useState(0)
  const [explain, setExplain] = useState(false)
  const [whatIfText, setWhatIfText] = useState('')
  const [whatIf, setWhatIf] = useState(null)
  const [action, run] = useAction()
  const [hoverClass, setHoverClass] = useState(null)
  const [benchmark, setBenchmark] = useState('SPY')
  const analysis = useApi(`/api/portfolio/${id}/analysis${explain ? '?explain=true' : ''}`, nonce)
  const profile = useApi(`/api/users/${id}`, nonce)
  const hasPortfolio = Boolean(analysis.data && !analysis.data.empty)
  const comparison = useApi(hasPortfolio && !whatIf ? `/api/portfolio/${id}/comparison?benchmark=${encodeURIComponent(benchmark)}` : null, nonce)
  const cmp = comparison.data
  const names = Object.keys(cmp?.comparison || {})

  const shown = whatIf || analysis.data
  const metrics = shown?.metrics
  const stocksPct = metrics ? Math.round((metrics.allocation_by_asset_class?.stock || 0) + (metrics.allocation_by_asset_class?.etf || 0)) : 0

  const changeRisk = (risk) => run(async () => {
    await api(`/api/users/${id}/profile`, { method: 'PUT', body: { risk_tolerance: risk } })
    setNonce((n) => n + 1)
  })
  const analyzeWhatIf = () => run(async () => {
    setWhatIf(await api(`/api/portfolio/${id}/analysis`, { method: 'POST', body: { holdings: parseHoldings(whatIfText), cash: 0 } }))
  })

  return (
    <div className="feature-grid portfolio-grid">
      <section className="panel portfolio-overview">
        <div className="panel-heading split">
          <div>
            <span className="pill pill-green">{whatIf ? 'What-if (not saved)' : 'Your saved portfolio'}</span>
            <h2>Portfolio snapshot</h2>
            <p>{profile.data?.synthetic === false ? 'Add holdings to see an analysis.' : 'A teaching example, not connected to a real account.'}</p>
          </div>
          {metrics && <div className="metric-right"><span>Total value</span><strong>{money(metrics.total_value)}</strong><StatusBadge status={shown.data_status} /></div>}
        </div>
        {analysis.loading && !shown && <Loading label="Analyzing portfolio…" />}
        <ErrorBox message={analysis.error} />
        {shown?.empty && <p className="callout-copy">This portfolio has no holdings yet. Try the what-if box to see how an analysis works.</p>}
        <DataNotices notices={shown?.notices} />
        {metrics && (
          <div className="allocation-block">
            <div>
              <div className="donut" style={{ background: donutBackground(metrics.allocation_by_asset_class, hoverClass) }} role="img"
                aria-label={`Allocation: ${Object.entries(metrics.allocation_by_asset_class).map(([k, v]) => `${v.toFixed(0)} percent ${k}`).join(', ')}`}>
                <div>
                  <strong>{hoverClass ? `${(metrics.allocation_by_asset_class[hoverClass] || 0).toFixed(0)}%` : `${stocksPct}%`}</strong>
                  <span>{hoverClass || 'stocks'}</span>
                </div>
              </div>
              <div className="class-bars" onMouseLeave={() => setHoverClass(null)}>
                {Object.entries(metrics.allocation_by_asset_class).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).map(([name, v]) => (
                  <div key={name} className={`class-row ${hoverClass && hoverClass !== name ? 'dim' : ''}`} tabIndex={0} title={`${name}: ${v.toFixed(1)}% of the portfolio`}
                    onMouseEnter={() => setHoverClass(name)} onFocus={() => setHoverClass(name)} onBlur={() => setHoverClass(null)}>
                    <span>{name}</span>
                    <div className="bar-track"><span className="bar-fill" style={{ width: `${Math.min(100, v)}%`, background: CLASS_COLORS[name] || 'var(--green)' }} /></div>
                    <b>{v.toFixed(0)}%</b>
                  </div>
                ))}
              </div>
            </div>
            <div className="holdings-list">
              {shown.holdings.map((h, i) => (
                <div className="holding" key={h.ticker}>
                  <span className={`legend-dot ${TONES[i % TONES.length]}`} />
                  <div><strong>{h.ticker}</strong><span>{h.shares} sh · {h.price_source}{h.price_status !== 'live' ? ` (${h.price_status})` : ''}</span></div>
                  <b>{pct(h.weight_pct, 0)}</b>
                  <span>{money(h.value)}</span>
                </div>
              ))}
              {metrics.cash_value > 0 && (
                <div className="holding"><span className="legend-dot coral" /><div><strong>Cash</strong><span>cash reserve</span></div>
                  <b>{pct((metrics.cash_value / metrics.total_value) * 100, 0)}</b><span>{money(metrics.cash_value)}</span></div>
              )}
            </div>
          </div>
        )}
        {metrics && (
          <ul className="notes-list">
            {metrics.notes.map((note) => <li key={note}>{note}</li>)}
            {metrics.weighted_expense_ratio != null && <li>Average fund cost: {metrics.weighted_expense_ratio.toFixed(2)}% a year (about {money(metrics.annual_fee_estimate)}).</li>}
            {metrics.unrealized_gain_loss != null && <li>Unrealized gain/loss: {money(metrics.unrealized_gain_loss)} ({pct(metrics.unrealized_gain_pct)}).</li>}
          </ul>
        )}
        {hasPortfolio && !whatIf && (
          <div className="compare-block">
            <div className="compare-head">
              <h3>How today&rsquo;s mix would have done vs a benchmark</h3>
              <select aria-label="Portfolio benchmark" value={benchmark} onChange={(e) => setBenchmark(e.target.value)}>
                {BENCHMARKS.map(([sym, name]) => <option key={sym} value={sym}>{name}</option>)}
              </select>
            </div>
            {comparison.loading && <Loading label="Comparing with the benchmark…" />}
            <ErrorBox message={comparison.error} />
            <DataNotices notices={cmp?.notices} />
            {cmp && names.length > 0 && (
              <>
                <div className="stat-chips">
                  <span>3 months vs {cmp.benchmark || 'benchmark'}: <b className={cmp.relative?.return_3m_diff_pts >= 0 ? 'positive' : 'negative'}>{pts(cmp.relative?.return_3m_diff_pts)}</b></span>
                  <span>Your mix: <b>{cmp.portfolio_trend?.return_3m_pct === null || cmp.portfolio_trend?.return_3m_pct === undefined ? '—' : `${cmp.portfolio_trend.return_3m_pct >= 0 ? '+' : ''}${cmp.portfolio_trend.return_3m_pct.toFixed(1)}%`}</b></span>
                  <span>Coverage: <b>{cmp.coverage_pct.toFixed(0)}%</b> of invested value</span>
                </div>
                <LineChart label={`Your current mix versus ${cmp.benchmark || 'the benchmark'}, each starting at 100`} format={(v) => v.toFixed(1)}
                  labels={cmp.comparison[names[0]].map((p) => shortDate(p.day))}
                  series={names.map((name) => ({ label: name, values: cmp.comparison[name].map((p) => p.value) }))} />
              </>
            )}
            {Object.keys(cmp?.excluded || {}).length > 0 && (
              <p className="callout-copy">Left out: {Object.entries(cmp.excluded).map(([t, why]) => `${t} (${why})`).join('; ')}.</p>
            )}
            {cmp?.insights?.length > 0 && (
              <div className="insight-box"><h3>What this shows</h3><ul className="insights">{cmp.insights.map((line) => <li key={line}>{line}</li>)}</ul></div>
            )}
          </div>
        )}
        {shown?.explanation && <div className="explanation-box"><Sparkles size={20} /><div><strong>In plain language</strong><p className="pre-wrap">{shown.explanation}</p></div></div>}
        {metrics && !whatIf && (
          <button className="button secondary" disabled={analysis.loading} onClick={() => setExplain(true)}>
            <Sparkles size={16} /> {explain ? 'Explained above' : 'Explain in plain language'}
          </button>
        )}
      </section>

      <aside className="side-stack">
        {metrics && (
          <section className="panel score-card">
            <div className="section-kicker"><ShieldCheck size={16} /> Diversification check</div>
            <div className="score-line"><strong>{Math.round(metrics.diversification_score)}</strong><span>/ 100</span><span className="pill pill-green">{metrics.risk_label} risk</span></div>
            <p>Spread across about {metrics.effective_positions.toFixed(1)} effective positions. Scores describe the mix, not future results.</p>
          </section>
        )}
        <section className="panel">
          <label className="select-label" htmlFor="risk-style">Comfort with risk</label>
          <div className="select-wrap">
            <select id="risk-style" value={profile.data?.risk_tolerance || ''} disabled={!profile.data || action.pending} onChange={(e) => changeRisk(e.target.value)}>
              {!profile.data && <option value="">—</option>}
              {RISKS.map((r) => <option key={r} value={r}>{r[0].toUpperCase() + r.slice(1)}</option>)}
            </select>
            <ChevronDown size={16} />
          </div>
          <p className="callout-copy">Used only to compare your mix with how much ups and downs you say you can tolerate.</p>
        </section>
        <section className="panel">
          <label className="select-label" htmlFor="what-if">What-if holdings (not saved)</label>
          <textarea id="what-if" className="what-if-input" rows={4} placeholder={'VTI 10\nBND 5'} value={whatIfText} onChange={(e) => setWhatIfText(e.target.value)} />
          <button className="button primary full" disabled={action.pending || !whatIfText.trim()} onClick={analyzeWhatIf}>Analyze what-if</button>
          {whatIf && <button className="text-button" onClick={() => setWhatIf(null)}><Check size={14} /> Back to my saved portfolio</button>}
          <ErrorBox message={action.error} />
        </section>
      </aside>
    </div>
  )
}
