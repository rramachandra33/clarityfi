import { useMemo, useState } from 'react'
import { BarChart3, Compass, Lightbulb, Search } from 'lucide-react'
import { money, pct, useApi } from '../api'
import { DataNotices, ErrorBox, LineChart, Loading, StatusBadge, movingAverage, providerName } from '../common'

const WATCHLIST = 'VTI,VXUS,BND,AAPL,MSFT'
const BENCHMARKS = [['SPY', 'S&P 500 (SPY)'], ['QQQ', 'Nasdaq-100 (QQQ)'], ['VTI', 'Total US market (VTI)'], ['BND', 'US bonds (BND)']]
const signed = (v, digits = 1) => (v === null || v === undefined ? '—' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(digits)}%`)
const shortDate = (iso) => new Date(`${iso}T00:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })

function ReturnBar({ label, value, scale }) {
  const width = value === null || value === undefined ? 0 : Math.min(50, (Math.abs(value) / scale) * 50)
  return (
    <div className="bar-row" title={`${label} return: ${signed(value)}`}>
      <span>{label}</span>
      <div className="bar-track"><span className={`bar-fill ${value >= 0 ? 'pos' : 'neg'}`} style={{ width: `${width}%` }} /></div>
      <b className={value >= 0 ? 'positive' : 'negative'}>{signed(value)}</b>
    </div>
  )
}

function Overview() {
  const overview = useApi('/api/market/overview')
  const segments = overview.data?.segments || []
  const scale = Math.max(5, ...segments.flatMap((s) => [Math.abs(s.return_1m_pct || 0), Math.abs(s.return_3m_pct || 0)]))
  return (
    <div className="overview-block">
      <h3>How the main markets have moved</h3>
      {overview.loading && <Loading label="Loading market overview…" />}
      <ErrorBox message={overview.error} />
      <DataNotices notices={overview.data?.notices} />
      <div className="segment-list">
        {segments.map((s) => (
          <div className="segment" key={s.symbol}>
            <div><b>{s.label}</b><small>{s.symbol} · {s.description}</small></div>
            {s.error ? <small>{s.error}</small> : (
              <div className="segment-bars"><ReturnBar label="1M" value={s.return_1m_pct} scale={scale} /><ReturnBar label="3M" value={s.return_3m_pct} scale={scale} /></div>
            )}
            <div>{!s.error && <><small>{s.trend_label}</small> <StatusBadge status={s.data_status} /></>}</div>
          </div>
        ))}
      </div>
      {overview.data?.takeaways?.length > 0 && <ul className="takeaways">{overview.data.takeaways.map((t) => <li key={t}>{t}</li>)}</ul>}
    </div>
  )
}

export default function MarketAnalysis() {
  const [symbols, setSymbols] = useState(WATCHLIST)
  const [draft, setDraft] = useState(WATCHLIST)
  const [selected, setSelected] = useState('VTI')
  const [benchmark, setBenchmark] = useState('SPY')
  const quotes = useApi(`/api/market/quotes?symbols=${encodeURIComponent(symbols)}`)
  const trend = useApi(selected ? `/api/market/trend/${encodeURIComponent(selected)}?days=180` : null)
  const context = useApi(selected ? `/api/market/context/${encodeURIComponent(selected)}?benchmark=${encodeURIComponent(benchmark)}` : null)
  const t = trend.data?.trend

  const priceSeries = useMemo(() => {
    const points = trend.data?.points || []
    const closes = points.map((p) => p.close)
    return { labels: points.map((p) => shortDate(p.day)), series: [
      { label: `${selected} close`, values: closes },
      { label: '20-day average', values: movingAverage(closes, 20), dashed: true },
      { label: '50-day average', values: movingAverage(closes, 50), dashed: true },
    ] }
  }, [trend.data, selected])

  const compare = context.data?.comparison || {}
  const names = Object.keys(compare)
  const compareLabels = names.length ? compare[names[0]].map((p) => shortDate(p.day)) : []
  const rel = context.data?.relative || {}

  return (
    <div className="feature-grid market-grid">
      <section className="panel market-main">
        <div className="panel-heading split">
          <div><span className="pill pill-yellow"><BarChart3 size={13} /> Market view</span><h2>Understand the day, not just the number</h2><p>Prices and history describe the past; they are not forecasts.</p></div>
        </div>

        <Overview />

        <form className="symbol-form" onSubmit={(e) => { e.preventDefault(); setSymbols(draft.trim() || WATCHLIST) }}>
          <input aria-label="Symbols" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="AAPL,MSFT,BTC-USD" maxLength={200} />
          <button className="button secondary" type="submit"><Search size={15} /> Update</button>
        </form>
        {quotes.loading && <Loading label="Loading quotes…" />}
        <ErrorBox message={quotes.error} />
        <DataNotices notices={quotes.data?.notices} />
        <div className="market-selector" role="tablist" aria-label="Quotes">
          {(quotes.data?.quotes || []).map((q) => (
            <button role="tab" aria-selected={selected === q.symbol} className={selected === q.symbol ? 'active' : ''} key={q.symbol} onClick={() => setSelected(q.symbol)}>
              {q.symbol} {money(q.price, 2)}
              {q.change_pct != null && <em className={q.change_pct >= 0 ? 'positive' : 'negative'}> {q.change_pct >= 0 ? '+' : ''}{q.change_pct.toFixed(2)}%</em>}
              <StatusBadge status={q.data_status} />
              <small className="provider-tag">via {providerName(q.source)}</small>
            </button>
          ))}
        </div>
        {Object.entries(quotes.data?.errors || {}).map(([s, msg]) => <p className="callout-copy" key={s}>{s}: {msg}</p>)}

        <div className="market-focus">
          <div>
            <span>{selected} · about 6 months{trend.data?.source ? ` · history via ${providerName(trend.data.source)}` : ''}</span>
            {t && <><strong className={t.return_3m_pct >= 0 ? 'positive' : 'negative'}>{t.trend_label}</strong>
              <p>3-month change {pct(t.return_3m_pct)} · 1-month {pct(t.return_1m_pct)} · volatility {pct(t.volatility_annual_pct)}</p></>}
            {trend.loading && <Loading label="Loading history…" />}
            <ErrorBox message={trend.error} />
          </div>
        </div>
        {trend.data && (
          <LineChart label={`${selected} closing prices with 20 and 50 day averages`} format={(v) => money(v, 2)} labels={priceSeries.labels} series={priceSeries.series} />
        )}
        <DataNotices notice={trend.data?.notice} />

        <div className="compare-head">
          <h3>{selected} compared with a benchmark</h3>
          <select aria-label="Benchmark" value={benchmark} onChange={(e) => setBenchmark(e.target.value)}>
            {BENCHMARKS.map(([sym, name]) => <option key={sym} value={sym}>{name}</option>)}
          </select>
        </div>
        {context.loading && <Loading label="Comparing…" />}
        <ErrorBox message={context.error} />
        <DataNotices notices={(context.data?.notices || []).filter((text) => text !== trend.data?.notice)} />
        {names.length > 1 && (
          <>
            <div className="stat-chips">
              <span>3 months vs {context.data.benchmark}: <b className={rel.return_3m_diff_pts >= 0 ? 'positive' : 'negative'}>{rel.return_3m_diff_pts === null || rel.return_3m_diff_pts === undefined ? '—' : `${rel.return_3m_diff_pts >= 0 ? '+' : ''}${rel.return_3m_diff_pts.toFixed(1)} pts`}</b></span>
              <span>1 month: <b className={rel.return_1m_diff_pts >= 0 ? 'positive' : 'negative'}>{rel.return_1m_diff_pts === null || rel.return_1m_diff_pts === undefined ? '—' : `${rel.return_1m_diff_pts >= 0 ? '+' : ''}${rel.return_1m_diff_pts.toFixed(1)} pts`}</b></span>
            </div>
            <LineChart label={`${selected} versus ${context.data.benchmark}, each starting at 100`} labels={compareLabels} format={(v) => v.toFixed(1)}
              series={names.map((name) => ({ label: name, values: compare[name].map((p) => p.value) }))} />
            <small className="callout-copy">Both lines start at 100 on the same day, so you can see how each would have grown.</small>
          </>
        )}
        {context.data?.insights?.length > 0 && (
          <div className="insight-box">
            <h3><Lightbulb size={14} /> What these numbers may mean</h3>
            <ul className="insights">{context.data.insights.map((line) => <li key={line}>{line}</li>)}</ul>
          </div>
        )}

        {t && (
          <div className="context-grid">
            <div><span>Moving averages</span><strong>{t.sma_20 ? money(t.sma_20, 2) : '—'} / {t.sma_50 ? money(t.sma_50, 2) : '—'}</strong><p>20- and 50-day averages smooth out daily noise.</p></div>
            <div><span>Largest drop</span><strong>{pct(t.max_drawdown_pct)}</strong><p>Peak-to-trough fall in this period.</p></div>
            <div><span>Range</span><strong>{money(t.period_low, 2)} – {money(t.period_high, 2)}</strong><p>Low and high close in this period.</p></div>
          </div>
        )}
        <p className="callout-copy"><Compass size={13} /> Hover over a chart (or focus it and use the arrow keys) to read exact values.</p>
      </section>
    </div>
  )
}
