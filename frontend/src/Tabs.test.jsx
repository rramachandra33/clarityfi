import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import PortfolioAnalysis from './tabs/Portfolio'
import MarketAnalysis from './tabs/Market'
import GoalPlanning from './tabs/Goals'
import NewsSynthesizer from './tabs/News'
import TaxEducation from './tabs/Tax'

const pdfMock = vi.hoisted(() => ({ save: vi.fn(), text: vi.fn() }))
vi.mock('jspdf', () => ({
  jsPDF: vi.fn(function () { return { setFontSize: vi.fn(), setTextColor: vi.fn(), text: pdfMock.text, addPage: vi.fn(), save: pdfMock.save } }),
}))

afterEach(() => { cleanup(); vi.unstubAllGlobals(); pdfMock.save.mockClear(); pdfMock.text.mockClear() })

const json = (data, status = 200, headers = {}) => ({ ok: status < 400, status, json: async () => data, headers: { get: (k) => headers[k] ?? null } })

// routes: [[substring, handler|data]] — first match wins
function stubFetch(routes) {
  const calls = []
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    calls.push({ url: String(url), method: options.method || 'GET', body: options.body })
    for (const [match, reply] of routes) {
      if (String(url).includes(match)) return typeof reply === 'function' ? reply(options) : json(reply)
    }
    return json({ detail: 'not stubbed', code: 'x' }, 500)
  }))
  return calls
}

const analysis = {
  user_id: 'learner-105', empty: false, what_if: false, data_status: 'stale', notice: 'n', notices: ['Prices for VTI are the last known values.'],
  metrics: { total_value: 60000, allocation_by_asset_class: { stock: 70, bond: 20, cash: 10 }, cash_value: 6000, diversification_score: 82, risk_label: 'Moderate',
    effective_positions: 3.2, notes: ['Broadly diversified.'], weighted_expense_ratio: 0.05, annual_fee_estimate: 30, unrealized_gain_loss: null },
  holdings: [{ ticker: 'VTI', shares: 10, value: 3000, weight_pct: 50, price_source: 'yFinance', price_status: 'stale' }],
}

describe('live tabs', () => {
  it('portfolio shows value, and discloses non-live prices', async () => {
    stubFetch([['/analysis', analysis], ['/api/users/', { risk_tolerance: 'moderate', synthetic: true }]])
    render(<PortfolioAnalysis userId="learner-105" />)
    expect(await screen.findByText('$60,000')).toBeInTheDocument()
    expect(screen.getByText(/last known values/)).toBeInTheDocument()
    expect(screen.getByText('Last known')).toBeInTheDocument()
    expect(screen.getByText('82')).toBeInTheDocument()
  })

  it('portfolio what-if posts parsed holdings and reports bad input', async () => {
    const calls = stubFetch([['/comparison', { user_id: 'learner-105', comparison: {}, insights: [], notices: [], excluded: {}, coverage_pct: 0 }], ['/analysis', (o) => json(o.method === 'POST' ? { ...analysis, what_if: true } : analysis)], ['/api/users/', { risk_tolerance: 'moderate' }]])
    render(<PortfolioAnalysis userId="learner-105" />)
    await screen.findByText('$60,000')
    fireEvent.change(screen.getByLabelText(/what-if holdings/i), { target: { value: 'not a holding line' } })
    fireEvent.click(screen.getByRole('button', { name: /analyze what-if/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/Could not read/)
    fireEvent.change(screen.getByLabelText(/what-if holdings/i), { target: { value: 'VTI 10\nBND 5' } })
    fireEvent.click(screen.getByRole('button', { name: /analyze what-if/i }))
    await screen.findByText(/What-if \(not saved\)/)
    expect(JSON.parse(calls.find((c) => c.method === 'POST').body).holdings).toEqual([{ ticker: 'VTI', shares: 10 }, { ticker: 'BND', shares: 5 }])
  })

  it('portfolio shows a friendly error when the service is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('network') }))
    render(<PortfolioAnalysis userId="learner-105" />)
    expect(await screen.findAllByText(/Could not reach the ClarityFI service/)).not.toHaveLength(0)
  })

  it('market labels sample quotes prominently', async () => {
    stubFetch([['/api/market/quotes', { quotes: [{ symbol: 'AAPL', price: 200, change_pct: 1.2, data_status: 'sample', source: 'mock' }], errors: { ZZZZ: 'Ticker was not found' },
      notices: ['AAPL is SAMPLE data, not a real price.'], data_status: 'sample' }],
      ['/api/market/trend', { symbol: 'AAPL', trend: { trend_label: 'Uptrend', return_3m_pct: 4, return_1m_pct: 1, volatility_annual_pct: 12, max_drawdown_pct: -5, period_low: 90, period_high: 110 },
        points: [{ day: '2026-01-01', close: 100 }, { day: '2026-01-02', close: 101 }], source: 'alphavantage', data_status: 'live' }]])
    render(<MarketAnalysis />)
    expect(await screen.findByText('SAMPLE DATA')).toBeInTheDocument()
    expect(screen.getByText(/not a real price/)).toBeInTheDocument()
    expect(screen.getByText(/ZZZZ: Ticker was not found/)).toBeInTheDocument()
    expect(await screen.findByText('Uptrend')).toBeInTheDocument()
    expect(screen.getByText('via Sample generator')).toBeInTheDocument()
    expect(await screen.findByText(/history via Alpha Vantage/)).toBeInTheDocument()
  })

  it('goals project, show scenarios, and save/delete', async () => {
    const projection = { goal: { target_amount: 80000 }, projection: { scenarios: [
      { label: 'Below expected', annual_return: 0.03, projected_value: 70000, required_monthly: 1000, on_track: false, shortfall: 10000 },
      { label: 'Expected', annual_return: 0.05, projected_value: 85000, required_monthly: 800, on_track: true, shortfall: 0 }],
      yearly_path: { a: [1, 2, 3], b: [1, 3, 5] } }, assumptions: ['Illustrative rates.'], explanation: null }
    let goals = [{ id: 7, goal: { name: 'Car', target_amount: 10000, target_date: '2029-01-01' } }]
    const calls = stubFetch([['/api/goals/project', projection], ['/api/goals/learner-105/7', () => { goals = []; return json(null, 204) }],
      ['/api/goals/learner-105', (o) => (o.method === 'POST' ? json({ id: 8 }, 201) : json(goals))]])
    render(<GoalPlanning userId="learner-105" />)
    expect(await screen.findByText('Car')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /^project my goal/i }))
    expect(await screen.findByText('Short by $10,000')).toBeInTheDocument()
    expect(screen.getByText('Illustrative rates.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /delete car/i }))
    await waitFor(() => expect(screen.getByText(/No saved goals yet/)).toBeInTheDocument())
    expect(calls.some((c) => c.method === 'DELETE')).toBe(true)
  })

  it('clicking a saved goal loads it into the builder and re-projects it', async () => {
    const future = new Date(Date.now() + 3 * 365.25 * 24 * 3600 * 1000).toISOString().slice(0, 10)
    const savedGoal = { name: 'Car', target_amount: 20000, target_date: future, current_savings: 5000, monthly_contribution: 300, assumed_return_pct: 4 }
    const projection = { goal: { target_amount: 20000 }, projection: { scenarios: [
      { label: 'Your assumption', annual_return: 0.04, projected_value: 21000, required_monthly: 400, on_track: true, shortfall: 0 }],
      yearly_path: { a: [1, 2, 3] } }, assumptions: ['Illustrative rates.'], explanation: null }
    const calls = stubFetch([['/api/goals/project', projection], ['/api/goals/learner-105', [{ id: 9, goal: savedGoal }]]])
    render(<GoalPlanning userId="learner-105" />)
    expect(await screen.findByText('Car')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /load saved goal car/i }))

    await waitFor(() => expect(screen.getByLabelText('Goal name')).toHaveValue('Car'))
    expect(screen.getByLabelText('Target amount')).toHaveValue(20000)
    expect(screen.getByLabelText('Already saved')).toHaveValue(5000)
    expect(screen.getByLabelText('Monthly contribution')).toHaveValue(300)
    expect(screen.getByLabelText('Assumed return')).toHaveValue(4)
    await screen.findByText('On track')

    const sent = JSON.parse(calls.find((c) => c.url === '/api/goals/project').body)
    expect(sent).toMatchObject({ name: 'Car', target_amount: 20000, current_savings: 5000, monthly_contribution: 300, assumed_return_pct: 4 })
    expect(sent.years).toBeGreaterThanOrEqual(2.5)
    expect(sent.years).toBeLessThanOrEqual(3.5)
  })

  it('goals offers a PDF download only when goals are saved, and it names the saved goal', async () => {
    stubFetch([['/api/goals/learner-105', [{ id: 7, goal: { name: 'Car', target_amount: 10000, target_date: '2029-01-01' } }]]])
    render(<GoalPlanning userId="learner-105" />)
    expect(await screen.findByText('Car')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /download pdf/i }))
    await waitFor(() => expect(pdfMock.save).toHaveBeenCalledWith('clarityfi-saved-goals.pdf'))
    expect(pdfMock.text.mock.calls.some(([text]) => text.includes('Car'))).toBe(true)
  })

  it('goals hides the PDF download when there are no saved goals', async () => {
    stubFetch([['/api/goals/learner-105', []]])
    render(<GoalPlanning userId="learner-105" />)
    await screen.findByText(/No saved goals yet/)
    expect(screen.queryByRole('button', { name: /download pdf/i })).not.toBeInTheDocument()
  })

  it('sends the assumed return, defaulting to 0, and renders every scenario the API returns', async () => {
    const threeScenarios = { goal: { target_amount: 80000 }, projection: { scenarios: [
      { label: 'Below your assumption', annual_return: -0.02, projected_value: 60000, required_monthly: 1200, on_track: false, shortfall: 20000 },
      { label: 'Your assumption', annual_return: 0.0, projected_value: 65000, required_monthly: 1100, on_track: false, shortfall: 15000 },
      { label: 'Above your assumption', annual_return: 0.02, projected_value: 70000, required_monthly: 1000, on_track: false, shortfall: 10000 }],
      yearly_path: { 'Below your assumption': [1, 2], 'Your assumption': [1, 2], 'Above your assumption': [1, 2] } },
      assumptions: ['Illustrative rates.'], explanation: null }
    const calls = stubFetch([['/api/goals/project', threeScenarios], ['/api/goals/learner-105', []]])
    render(<GoalPlanning userId="learner-105" />)
    expect(await screen.findByLabelText('Assumed return')).toHaveValue(0)
    fireEvent.click(screen.getByRole('button', { name: /^project my goal/i }))
    await screen.findAllByText('Below your assumption')
    expect(screen.getAllByText('Your assumption').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Above your assumption').length).toBeGreaterThan(0)
    expect(JSON.parse(calls.find((c) => c.url === '/api/goals/project').body).assumed_return_pct).toBe(0)

    fireEvent.change(screen.getByLabelText('Assumed return'), { target: { value: '6' } })
    fireEvent.click(screen.getByRole('button', { name: /^project my goal/i }))
    await waitFor(() => expect(JSON.parse(calls[calls.length - 1].body).assumed_return_pct).toBe(6))
  })

  it('goals surface validation errors from the API', async () => {
    stubFetch([['/api/goals/project', () => json({ detail: 'Provide exactly one of target_date or years', code: 'x' }, 422)], ['/api/goals/', []]])
    render(<GoalPlanning userId="learner-105" />)
    fireEvent.click(screen.getByRole('button', { name: /^project my goal/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/exactly one/)
  })

  it('news shows sample notice, safe links, and handles 429', async () => {
    let n = 0
    stubFetch([['/api/news', () => (n++ === 0
      ? json({ items: [{ title: 'Rates hold', url: 'https://ex.com/a', source: 'Wire' }, { title: 'Bad link', url: 'javascript:alert(1)' }], data_status: 'sample', notice: 'These are SAMPLE headlines, not real news.' })
      : json({ detail: 'x', code: 'too_many_requests' }, 429, { 'Retry-After': '30' }))]])
    render(<NewsSynthesizer />)
    fireEvent.click(screen.getByRole('button', { name: /^search$/i }))
    expect(await screen.findByText(/SAMPLE headlines, not real news/)).toBeInTheDocument()
    expect(screen.getByText('SAMPLE DATA')).toBeInTheDocument()
    expect(screen.getAllByRole('link')).toHaveLength(1)
    fireEvent.click(screen.getByRole('button', { name: /^search$/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/wait 30 seconds/)
  })

  it('tax asks the assistant and shows the answer with sources and notice', async () => {
    const calls = stubFetch([['/api/chat', { answer: 'A Roth IRA is funded with after-tax money.', notice: 'Answered from a saved response.',
      sources: [{ index: 1, title: 'Roth IRA', source: 'IRS', url: 'https://irs.gov/roth' }] }]])
    render(<TaxEducation userId="learner-105" />)
    fireEvent.change(screen.getByLabelText('Tax question'), { target: { value: 'How does a Roth IRA work?' } })
    fireEvent.click(screen.getByRole('button', { name: /^ask$/i }))
    expect(await screen.findByText(/after-tax money/)).toBeInTheDocument()
    expect(screen.getByText(/saved response/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Roth IRA' })).toHaveAttribute('href', 'https://irs.gov/roth')
    expect(JSON.parse(calls[0].body).question).toBe('How does a Roth IRA work?')
  })
})

describe('StrictMode', () => {
  it('finishes pending state after the answer arrives (dev double-mount must not freeze the UI)', async () => {
    const { StrictMode } = await import('react')
    stubFetch([['/api/chat', { answer: 'A Roth IRA is funded with after-tax money.', sources: [] }]])
    render(<StrictMode><TaxEducation userId="learner-105" /></StrictMode>)
    fireEvent.change(screen.getByLabelText('Tax question'), { target: { value: 'How does a Roth IRA work?' } })
    fireEvent.click(screen.getByRole('button', { name: /^ask$/i }))
    await screen.findByText(/after-tax money/)
    await waitFor(() => expect(screen.queryByText(/Looking that up/)).not.toBeInTheDocument())
  })
})
