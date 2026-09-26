import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LineChart, movingAverage } from './common'
import MarketAnalysis from './tabs/Market'
import GoalPlanning from './tabs/Goals'
import PortfolioAnalysis from './tabs/Portfolio'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

const json = (data, status = 200) => ({ ok: status < 400, status, json: async () => data, headers: { get: () => null } })
function stubFetch(routes) {
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    for (const [match, reply] of routes) if (String(url).includes(match)) return typeof reply === 'function' ? reply(options) : json(reply)
    return json({ detail: 'not stubbed', code: 'x' }, 500)
  }))
}

describe('LineChart', () => {
  const series = [{ label: 'Price', values: [10, 20, 15, 30] }, { label: 'Average', values: [null, null, 15, 18], dashed: true }]
  const labels = ['Jan 1', 'Jan 2', 'Jan 3', 'Jan 4']

  it('shows a tooltip with every series value on hover and hides it on leave', () => {
    render(<LineChart series={series} labels={labels} label="Test chart" format={(v) => `$${v}`} />)
    const box = screen.getByRole('group', { name: /test chart/i })
    box.getBoundingClientRect = () => ({ left: 0, width: 300, top: 0, height: 100 })
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    fireEvent.mouseMove(box, { clientX: 300 })
    const tip = screen.getByRole('status')
    expect(tip).toHaveTextContent('Jan 4')
    expect(tip).toHaveTextContent('Price: $30')
    expect(tip).toHaveTextContent('Average: $18')
    fireEvent.mouseMove(box, { clientX: 0 })
    expect(screen.getByRole('status')).toHaveTextContent('Jan 1')
    expect(screen.getByRole('status')).not.toHaveTextContent('Average')  // gap values are skipped
    fireEvent.mouseLeave(box)
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })

  it('is keyboard accessible and labels the axis, legend and reference line', () => {
    render(<LineChart series={series} labels={labels} label="Test chart" format={(v) => `$${v}`} refLine={{ value: 25, label: 'Target $25' }} />)
    const box = screen.getByRole('group', { name: /arrow keys/i })
    fireEvent.keyDown(box, { key: 'ArrowRight' })
    expect(screen.getByRole('status')).toHaveTextContent('Jan 1')
    fireEvent.keyDown(box, { key: 'ArrowRight' })
    fireEvent.keyDown(box, { key: 'ArrowLeft' })
    expect(screen.getByRole('status')).toHaveTextContent('Jan 1')
    fireEvent.keyDown(box, { key: 'Escape' })
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(screen.getByText('Target $25')).toBeInTheDocument()
    expect(screen.getAllByText('Jan 1').length).toBeGreaterThan(0)
    expect(screen.getByText(/Low \$10/)).toBeInTheDocument()
  })

  it('renders nothing for fewer than two points, and computes moving averages', () => {
    const { container } = render(<LineChart series={[{ label: 'x', values: [1] }]} label="x" />)
    expect(container).toBeEmptyDOMElement()
    expect(movingAverage([1, 2, 3, 4], 2)).toEqual([null, 1.5, 2.5, 3.5])
  })
})

const trendBody = { symbol: 'VTI', trend: { trend_label: 'Uptrend', return_3m_pct: 4, return_1m_pct: 1, volatility_annual_pct: 12, max_drawdown_pct: -5, period_low: 90, period_high: 110, sma_20: 100, sma_50: 98 },
  points: Array.from({ length: 60 }, (_, i) => ({ day: `2026-03-${String((i % 28) + 1).padStart(2, '0')}`, close: 100 + i })), source: 'yfinance', data_status: 'live' }

describe('Market context', () => {
  it('shows the segment overview with takeaways, the benchmark comparison and the plain-language insights', async () => {
    stubFetch([
      ['/api/market/overview', { segments: [
        { symbol: 'SPY', label: 'US large companies', description: 'S&P 500 proxy', return_1m_pct: 2, return_3m_pct: 6, trend_label: 'Uptrend', data_status: 'live' },
        { symbol: 'BND', label: 'US bonds', description: 'bonds', return_1m_pct: -0.5, return_3m_pct: -1, trend_label: 'Downtrend', data_status: 'stale' },
        { symbol: 'IWM', label: 'US small companies', description: 'x', error: 'Data unavailable right now' }],
        notices: ['BND is from a saved copy.'], takeaways: ['Over 3 months, US large companies led (+6.0%).'] }],
      ['/api/market/context', { symbol: 'VTI', benchmark: 'SPY', trend: trendBody.trend, relative: { return_3m_diff_pts: -1.5, return_1m_diff_pts: 0.4 },
        insights: ['The price is above its 20-day and 50-day averages.', 'This describes the past only.'], notices: [],
        comparison: { VTI: [{ day: '2026-03-01', value: 100 }, { day: '2026-03-02', value: 101 }], SPY: [{ day: '2026-03-01', value: 100 }, { day: '2026-03-02', value: 102 }] } }],
      ['/api/market/quotes', { quotes: [{ symbol: 'VTI', price: 300, change_pct: 1, data_status: 'live', source: 'yfinance' }], errors: {}, notices: [] }],
      ['/api/market/trend', trendBody],
    ])
    render(<MarketAnalysis />)
    expect(await screen.findByText('How the main markets have moved')).toBeInTheDocument()
    expect(await screen.findByText(/US large companies led/)).toBeInTheDocument()
    expect(screen.getByText('BND is from a saved copy.')).toBeInTheDocument()
    expect(screen.getByText('Last known')).toBeInTheDocument()
    expect(screen.getByText('Data unavailable right now')).toBeInTheDocument()
    expect(await screen.findByText(/What these numbers may mean/)).toBeInTheDocument()
    expect(screen.getByText('This describes the past only.')).toBeInTheDocument()
    expect(screen.getByText('-1.5 pts')).toBeInTheDocument()
    expect(screen.getByRole('group', { name: /VTI versus SPY/i })).toBeInTheDocument()
    expect(screen.getByRole('group', { name: /20 and 50 day averages/i })).toBeInTheDocument()
  })

  it('asks for a different benchmark when the selector changes', async () => {
    const seen = []
    stubFetch([['/api/market/overview', { segments: [], notices: [], takeaways: [] }], ['/api/market/quotes', { quotes: [], errors: {}, notices: [] }],
      ['/api/market/trend', trendBody], ['/api/market/context', (o) => json({ symbol: 'VTI', trend: trendBody.trend, insights: [], comparison: {}, notices: [], relative: {} })]])
    const original = globalThis.fetch
    vi.stubGlobal('fetch', vi.fn(async (url, o) => { seen.push(String(url)); return original(url, o) }))
    render(<MarketAnalysis />)
    fireEvent.change(await screen.findByLabelText('Benchmark'), { target: { value: 'QQQ' } })
    await waitFor(() => expect(seen.some((u) => u.includes('/context/VTI?benchmark=QQQ'))).toBe(true))
  })
})

describe('Goals and Portfolio visuals', () => {
  it('draws the goal target as a labelled reference line with year labels', async () => {
    stubFetch([
      ['/api/goals/project', { goal: { target_amount: 80000 }, projection: { scenarios: [{ label: 'Expected', annual_return: 0.05, projected_value: 85000, required_monthly: 800, on_track: true, shortfall: 0 }],
        yearly_path: { 'Below expectations': [1, 2, 3], Expected: [1, 3, 5] } }, assumptions: [], explanation: null }],
      ['/api/goals/learner-105', []],
    ])
    render(<GoalPlanning userId="learner-105" />)
    fireEvent.click(screen.getByRole('button', { name: /^project my goal/i }))
    expect(await screen.findByText('Target $80,000')).toBeInTheDocument()
    const box = screen.getByRole('group', { name: /with the target/i })
    box.getBoundingClientRect = () => ({ left: 0, width: 200, top: 0, height: 100 })
    fireEvent.mouseMove(box, { clientX: 200 })
    expect(screen.getByRole('status')).toHaveTextContent('Year 2')
    expect(screen.getByRole('status')).toHaveTextContent('Expected: $5')
  })

  it('highlights an asset class when its bar is hovered', async () => {
    stubFetch([
      ['/analysis', { user_id: 'learner-105', empty: false, data_status: 'live', notices: [], metrics: { total_value: 1000, allocation_by_asset_class: { stock: 70, bond: 20, cash: 10 },
        cash_value: 100, diversification_score: 80, risk_label: 'Moderate', effective_positions: 3, notes: [], weighted_expense_ratio: null }, holdings: [] }],
      ['/api/users/', { risk_tolerance: 'moderate', synthetic: true }],
    ])
    render(<PortfolioAnalysis userId="learner-105" />)
    const bond = (await screen.findByText('bond')).closest('.class-row')
    expect(document.querySelector('.donut')).toHaveTextContent('70%')
    fireEvent.mouseEnter(bond)
    expect(document.querySelector('.donut')).toHaveTextContent('20%')
    expect(document.querySelector('.donut')).toHaveTextContent('bond')
    fireEvent.mouseLeave(bond.parentElement)
    expect(document.querySelector('.donut')).toHaveTextContent('70%')
  })
})

describe('Portfolio vs benchmark', () => {
  const analysisBody = { user_id: 'learner-105', empty: false, data_status: 'live', notices: [], holdings: [],
    metrics: { total_value: 1000, allocation_by_asset_class: { stock: 100 }, cash_value: 0, diversification_score: 80, risk_label: 'Moderate', effective_positions: 3, notes: [], weighted_expense_ratio: null } }
  const cmpBody = { user_id: 'learner-105', benchmark: 'SPY', included: ['VTI'], excluded: { XYZ: 'Price history unavailable' }, coverage_pct: 92,
    portfolio_trend: { return_3m_pct: 5.1 }, relative: { return_3m_diff_pts: 1.2 }, notices: ['Cached price history from Yahoo Finance, about 5 minutes old. This is not a fresh lookup.'],
    insights: ['Had you held today\'s mix for the last 3 months, it would have returned +5.1%.', 'This replays your current holdings over past prices.'],
    comparison: { 'Your mix': [{ day: '2026-03-01', value: 100 }, { day: '2026-03-02', value: 101 }], SPY: [{ day: '2026-03-01', value: 100 }, { day: '2026-03-02', value: 100.5 }] } }

  it('shows the chart, gap, coverage, exclusions, notices and insights', async () => {
    stubFetch([['/comparison', cmpBody], ['/analysis', analysisBody], ['/api/users/', { risk_tolerance: 'moderate', synthetic: true }]])
    render(<PortfolioAnalysis userId="learner-105" />)
    expect(await screen.findByText(/How today.s mix would have done/)).toBeInTheDocument()
    expect(await screen.findByText('+1.2 pts')).toBeInTheDocument()
    expect(screen.getByText('92%')).toBeInTheDocument()
    expect(screen.getByText(/Left out: XYZ \(Price history unavailable\)/)).toBeInTheDocument()
    expect(screen.getByText(/not a fresh lookup/)).toBeInTheDocument()
    expect(screen.getByText(/This replays your current holdings/)).toBeInTheDocument()
    expect(screen.getByRole('group', { name: /current mix versus SPY/i })).toBeInTheDocument()
  })

  it('reloads with the chosen benchmark, and stays hidden for an empty portfolio', async () => {
    const seen = []
    stubFetch([['/comparison', cmpBody], ['/analysis', analysisBody], ['/api/users/', { risk_tolerance: 'moderate' }]])
    const original = globalThis.fetch
    vi.stubGlobal('fetch', vi.fn(async (url, o) => { seen.push(String(url)); return original(url, o) }))
    render(<PortfolioAnalysis userId="learner-105" />)
    fireEvent.change(await screen.findByLabelText('Portfolio benchmark'), { target: { value: 'QQQ' } })
    await waitFor(() => expect(seen.some((u) => u.includes('/comparison?benchmark=QQQ'))).toBe(true))
    cleanup()
    stubFetch([['/analysis', { user_id: 'x', empty: true }], ['/api/users/', { risk_tolerance: 'moderate' }]])
    render(<PortfolioAnalysis userId="learner-110" />)
    await screen.findByText(/no holdings yet/)
    expect(screen.queryByText(/How today.s mix would have done/)).not.toBeInTheDocument()
  })
})
