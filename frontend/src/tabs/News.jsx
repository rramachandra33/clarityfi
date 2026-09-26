import { useState } from 'react'
import { ExternalLink, Newspaper, Search, Sparkles } from 'lucide-react'
import { api, useAction } from '../api'
import { DataNotices, ErrorBox, Loading, StatusBadge } from '../common'

export default function NewsSynthesizer() {
  const [query, setQuery] = useState('interest rates')
  const [data, setData] = useState(null)
  const [action, run] = useAction()
  const search = (summarize) => run(async () => {
    setData(await api(`/api/news?q=${encodeURIComponent(query)}&limit=6&summarize=${summarize}`))
  })

  return (
    <div className="feature-grid news-grid">
      <section className="panel news-panel">
        <div className="panel-heading split">
          <div><span className="pill pill-purple"><Newspaper size={13} /> Headlines</span><h2>See the meaning behind the news</h2><p>Headlines are third-party reports, not verified facts. Open the source to read the full story.</p></div>
          {data && <StatusBadge status={data.data_status} />}
        </div>
        <form className="symbol-form" onSubmit={(e) => { e.preventDefault(); search(false) }}>
          <input aria-label="News topic" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Company, ticker or topic" maxLength={200} />
          <button className="button secondary" type="submit" disabled={action.pending}><Search size={15} /> Search</button>
          <button className="button primary" type="button" disabled={action.pending} onClick={() => search(true)}><Sparkles size={15} /> Search and summarize</button>
        </form>
        {action.pending && <Loading label="Searching…" />}
        <ErrorBox message={action.error} />
        {!data && !action.pending && <p className="callout-copy">Search a topic to see recent headlines.</p>}
        {data && <DataNotices notice={data.notice} />}
        {data?.summary && <div className="explanation-box"><Sparkles size={20} /><div><strong>Summary</strong><p className="pre-wrap">{data.summary}</p></div></div>}
        {data && data.items.length === 0 && <p className="callout-copy">No recent headlines found. Try a company name or a topic such as “inflation”.</p>}
        <div className="news-list">
          {data?.items.map((item) => (
            <article className="news-item" key={item.url + item.title}>
              <div>
                <span className="news-meta">{[item.source, item.published?.replace(/,?\s*\+0000 UTC$/, '')].filter(Boolean).join(' · ')}</span>
                <h3>{item.title}</h3>
                {item.snippet && <p>{item.snippet}</p>}
                {/^https?:\/\//.test(item.url) && <a className="source-link" href={item.url} target="_blank" rel="noreferrer noopener"><ExternalLink size={13} /> Read at source</a>}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
