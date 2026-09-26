import { useState } from 'react'
import { Info } from 'lucide-react'

const PROVIDERS = { alphavantage: 'Alpha Vantage', yfinance: 'Yahoo Finance (yFinance)', mock: 'Sample generator', mock_sample: 'Sample generator', sample: 'Sample generator' }
export const providerName = (source) => (source ? PROVIDERS[source] || source : null)

const LABELS = { live: 'Live', cached: 'Saved copy', stale: 'Last known', sample: 'SAMPLE DATA' }

// Every non-live value is disclosed: a badge, plus the plain-language notice(s) from the API.
export function StatusBadge({ status }) {
  if (!status || status === 'live') return null
  return <span className={`status-badge status-${status}`}>{LABELS[status] || status}</span>
}

export function DataNotices({ notices, notice }) {
  const items = [...new Set([...(notices || []), ...(notice ? [notice] : [])])].filter(Boolean)
  if (!items.length) return null
  return (
    <div className="data-notice" role="note">
      <Info size={14} />
      <div>{items.map((text) => <div key={text}>{text}</div>)}</div>
    </div>
  )
}

export function ErrorBox({ message }) {
  return message ? <div className="error-box" role="alert">{message}</div> : null
}

export function Loading({ label = 'Loading…' }) {
  return <p className="loading-line" role="status">{label}</p>
}

export const CHART_COLORS = ['#176f59', '#e4b04a', '#6651a8', '#c4553f', '#2f6f9e', '#8a9a3a']

const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n))

// Interactive SVG line chart. series = [{ label, values (null = gap), color?, dashed? }], labels = x label per index.
// Hover (or arrow keys when focused) shows a crosshair, a dot per series and a tooltip; legend and axis labels included.
export function LineChart({ series, labels = [], height = 170, label, format = (v) => v, refLine = null, legend = true }) {
  const [hover, setHover] = useState(null)
  const n = Math.max(0, ...series.map((s) => s.values.length))
  if (n < 2) return null
  const width = 600
  const pad = 10
  const nums = series.flatMap((s) => s.values.filter((v) => v !== null && v !== undefined))
  if (refLine) nums.push(refLine.value)
  const min = Math.min(...nums)
  const max = Math.max(...nums)
  const span = max - min || 1
  const x = (i) => (i / (n - 1)) * width
  const y = (v) => height - pad - ((v - min) / span) * (height - pad * 2)
  const pathFor = (values) => {
    let d = ''
    let pen = false
    values.forEach((v, i) => {
      if (v === null || v === undefined) { pen = false; return }
      d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)} `
      pen = true
    })
    return d.trim()
  }
  const colorOf = (s, i) => s.color || CHART_COLORS[i % CHART_COLORS.length]
  const move = (event) => {
    const box = event.currentTarget.getBoundingClientRect()
    if (!box.width) return
    setHover(clamp(Math.round(((event.clientX - box.left) / box.width) * (n - 1)), 0, n - 1))
  }
  const key = (event) => {
    if (event.key === 'ArrowRight') { event.preventDefault(); setHover((h) => clamp((h ?? -1) + 1, 0, n - 1)) }
    else if (event.key === 'ArrowLeft') { event.preventDefault(); setHover((h) => clamp((h ?? n) - 1, 0, n - 1)) }
    else if (event.key === 'Escape') setHover(null)
  }
  const mid = Math.floor((n - 1) / 2)
  const tipLeft = hover === null ? 0 : (hover / (n - 1)) * 100
  return (
    <figure className="line-chart">
      {legend && series.length > 1 && (
        <div className="chart-legend">
          {series.map((s, i) => <span key={s.label}><i style={{ background: colorOf(s, i) }} />{s.label}</span>)}
          {refLine && <span><i className="ref" />{refLine.label}</span>}
        </div>
      )}
      <div className="chart-box" tabIndex={0} role="group" aria-label={`${label}. Use the left and right arrow keys to read values.`}
        onMouseMove={move} onMouseLeave={() => setHover(null)} onKeyDown={key} onBlur={() => setHover(null)}>
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
          {refLine && <line x1="0" x2={width} y1={y(refLine.value)} y2={y(refLine.value)} className="ref-line" />}
          {series.map((s, i) => <path key={s.label} d={pathFor(s.values)} className="line" style={{ stroke: colorOf(s, i), strokeDasharray: s.dashed ? '5 4' : undefined }} fill="none" />)}
          {hover !== null && <line x1={x(hover)} x2={x(hover)} y1="0" y2={height} className="crosshair" />}
        </svg>
        {hover !== null && series.map((s, i) => {
          const v = s.values[hover]
          return v === null || v === undefined ? null
            : <i key={s.label} className="chart-dot" style={{ left: `${tipLeft}%`, top: `${(y(v) / height) * 100}%`, background: colorOf(s, i) }} />
        })}
        {hover !== null && (
          <div className="chart-tip" role="status" style={{ left: `${tipLeft}%`, transform: `translateX(${tipLeft > 60 ? '-105%' : '5%'})` }}>
            <strong>{labels[hover] ?? `#${hover + 1}`}</strong>
            {series.map((s, i) => (s.values[hover] === null || s.values[hover] === undefined ? null
              : <span key={s.label}><i style={{ background: colorOf(s, i) }} />{s.label}: {format(s.values[hover])}</span>))}
          </div>
        )}
      </div>
      <figcaption className="chart-axis">
        <span>{labels[0] ?? format(min)}</span><span>{labels[mid] ?? ''}</span><span>{labels[n - 1] ?? format(max)}</span>
      </figcaption>
      <div className="chart-range"><span>Low {format(min)}</span><span>High {format(max)}</span></div>
    </figure>
  )
}

// Simple moving average series aligned to `values` (null until enough data).
export function movingAverage(values, window) {
  return values.map((_, i) => (i + 1 < window ? null : values.slice(i + 1 - window, i + 1).reduce((a, b) => a + b, 0) / window))
}
