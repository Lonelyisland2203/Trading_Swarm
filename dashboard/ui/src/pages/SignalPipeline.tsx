import { useEffect, useState } from 'react'
import { ChevronDown, ChevronRight, Check, X } from 'lucide-react'
import { clsx } from 'clsx'
import { getRecentSignals } from '@/api/client'
import type { Signal } from '@/types'

export function SignalPipeline() {
  const [signals, setSignals] = useState<Signal[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedRow, setExpandedRow] = useState<string | null>(null)

  const fetchData = async () => {
    try {
      const data = await getRecentSignals()
      setSignals(data)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch signals')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return <LoadingState />
  }

  if (error) {
    return <ErrorState message={error} onRetry={fetchData} />
  }

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-medium text-[var(--text-secondary)]">
          Recent Signals (Last 50)
        </h2>
        <span className="text-xs text-[var(--text-muted)]">
          Polling every 30s
        </span>
      </div>

      <div className="overflow-auto rounded border border-[var(--border)]" style={{ maxHeight: 'calc(100vh - 200px)' }}>
        <table className="w-full border-collapse">
          <thead className="sticky top-0 bg-[var(--bg-surface)]">
            <tr>
              <th className="w-8 px-2 py-2" />
              <th className="px-3 py-2 text-xs uppercase tracking-wider text-[var(--text-muted)] text-left">
                Timestamp
              </th>
              <th className="px-3 py-2 text-xs uppercase tracking-wider text-[var(--text-muted)] text-left">
                Symbol
              </th>
              <th className="px-3 py-2 text-xs uppercase tracking-wider text-[var(--text-muted)] text-right">
                XGB Prob
              </th>
              <th className="px-3 py-2 text-xs uppercase tracking-wider text-[var(--text-muted)] text-center">
                XGB Dir
              </th>
              <th className="px-3 py-2 text-xs uppercase tracking-wider text-[var(--text-muted)] text-center">
                Regime
              </th>
              <th className="px-3 py-2 text-xs uppercase tracking-wider text-[var(--text-muted)] text-center">
                Synthesis
              </th>
              <th className="px-3 py-2 text-xs uppercase tracking-wider text-[var(--text-muted)] text-center">
                Veto
              </th>
              <th className="px-3 py-2 text-xs uppercase tracking-wider text-[var(--text-muted)] text-right">
                Outcome
              </th>
            </tr>
          </thead>
          <tbody>
            {signals.length === 0 ? (
              <tr>
                <td colSpan={9} className="px-3 py-8 text-center text-[var(--text-muted)]">
                  No signals recorded
                </td>
              </tr>
            ) : (
              signals.map((signal) => {
                const key = `${signal.timestamp}-${signal.symbol}`
                const isExpanded = expandedRow === key
                const rowBg = getRowBackground(signal)

                return (
                  <>
                    <tr
                      key={key}
                      onClick={() => setExpandedRow(isExpanded ? null : key)}
                      className={clsx(
                        'cursor-pointer border-b border-[var(--border-subtle)] hover:bg-[var(--bg-elevated)]',
                        rowBg
                      )}
                    >
                      <td className="px-2 py-2 text-[var(--text-muted)]">
                        {signal.llm_context ? (
                          isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />
                        ) : null}
                      </td>
                      <td className="px-3 py-2 text-sm font-mono text-[var(--text-secondary)]">
                        {formatTimestamp(signal.timestamp)}
                      </td>
                      <td className="px-3 py-2 text-sm font-mono font-medium">
                        {signal.symbol}
                      </td>
                      <td className="px-3 py-2 text-sm font-mono text-right">
                        <ProbabilityCell value={signal.xgboost_prob} />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <DirectionBadge direction={signal.xgboost_direction} />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <RegimeBadge regime={signal.llm_context?.regime_flag} />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <DirectionBadge direction={signal.synthesis_decision} />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <VetoIndicator veto={signal.veto} />
                      </td>
                      <td className="px-3 py-2 text-sm font-mono text-right">
                        <OutcomeCell signal={signal} />
                      </td>
                    </tr>
                    {isExpanded && signal.llm_context && (
                      <tr key={`${key}-expanded`} className="bg-[var(--bg-elevated)]">
                        <td colSpan={9} className="px-6 py-4">
                          <ExpandedContext context={signal.llm_context} />
                        </td>
                      </tr>
                    )}
                  </>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ProbabilityCell({ value }: { value?: number }) {
  if (value === undefined) return <span className="text-[var(--text-muted)]">--</span>

  const pct = (value * 100).toFixed(1)
  const intensity = Math.min(value * 1.5, 1) // Scale for visual effect

  return (
    <span
      style={{
        color: value >= 0.55
          ? `rgba(0, 208, 132, ${intensity})`
          : `rgba(139, 148, 158, ${intensity})`,
      }}
    >
      {pct}%
    </span>
  )
}

function DirectionBadge({ direction }: { direction?: string }) {
  if (!direction) return <span className="text-[var(--text-muted)]">--</span>

  const colorMap: Record<string, string> = {
    LONG: 'bg-[var(--green)] text-[var(--bg-primary)]',
    SHORT: 'bg-[var(--red)] text-white',
    FLAT: 'bg-[var(--text-muted)] text-[var(--bg-primary)]',
  }

  return (
    <span className={clsx('px-2 py-0.5 text-xs font-medium rounded', colorMap[direction] || 'bg-[var(--border)]')}>
      {direction}
    </span>
  )
}

function RegimeBadge({ regime }: { regime?: string }) {
  if (!regime) return <span className="text-[var(--text-muted)]">--</span>

  const colorMap: Record<string, string> = {
    BULLISH: 'text-[var(--green)] border-[var(--green)]',
    BEARISH: 'text-[var(--red)] border-[var(--red)]',
    NEUTRAL: 'text-[var(--text-muted)] border-[var(--border)]',
    VOLATILE: 'text-[var(--amber)] border-[var(--amber)]',
  }

  return (
    <span
      className={clsx(
        'px-2 py-0.5 text-xs border rounded',
        colorMap[regime] || 'border-[var(--border)] text-[var(--text-muted)]'
      )}
    >
      {regime}
    </span>
  )
}

function VetoIndicator({ veto }: { veto?: boolean }) {
  if (veto === undefined) return <span className="text-[var(--text-muted)]">--</span>
  return veto ? (
    <X className="w-4 h-4 text-[var(--red)] mx-auto" />
  ) : (
    <Check className="w-4 h-4 text-[var(--green)] mx-auto" />
  )
}

function OutcomeCell({ signal }: { signal: Signal }) {
  if (!signal.outcome_resolved) {
    return <span className="text-[var(--amber)]">pending</span>
  }
  if (signal.outcome_pnl === null || signal.outcome_pnl === undefined) {
    return <span className="text-[var(--text-muted)]">--</span>
  }
  const pnl = signal.outcome_pnl
  return (
    <span className={pnl >= 0 ? 'text-[var(--green)]' : 'text-[var(--red)]'}>
      {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
    </span>
  )
}

function ExpandedContext({ context }: { context: Signal['llm_context'] }) {
  if (!context) return null

  return (
    <div className="grid grid-cols-2 gap-4 text-sm">
      <div>
        <div className="text-xs uppercase tracking-wider text-[var(--green)] mb-2">
          Bullish Factors
        </div>
        <ul className="space-y-1 text-[var(--text-secondary)]">
          {context.bullish_factors.length > 0 ? (
            context.bullish_factors.map((f, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-[var(--green)]">+</span>
                <span>{f}</span>
              </li>
            ))
          ) : (
            <li className="text-[var(--text-muted)]">None identified</li>
          )}
        </ul>
      </div>
      <div>
        <div className="text-xs uppercase tracking-wider text-[var(--red)] mb-2">
          Bearish Factors
        </div>
        <ul className="space-y-1 text-[var(--text-secondary)]">
          {context.bearish_factors.length > 0 ? (
            context.bearish_factors.map((f, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-[var(--red)]">-</span>
                <span>{f}</span>
              </li>
            ))
          ) : (
            <li className="text-[var(--text-muted)]">None identified</li>
          )}
        </ul>
      </div>
    </div>
  )
}

function getRowBackground(signal: Signal): string {
  if (!signal.outcome_resolved) return ''
  if (signal.outcome_pnl === null || signal.outcome_pnl === undefined) return ''
  return signal.outcome_pnl >= 0 ? 'bg-[var(--green-dim)]' : 'bg-[var(--red-dim)]'
}

function formatTimestamp(ts: string): string {
  const date = new Date(ts)
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}

function LoadingState() {
  return (
    <div className="p-6 flex items-center justify-center h-64">
      <div className="text-[var(--text-muted)]">Loading signals...</div>
    </div>
  )
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="p-6 flex flex-col items-center justify-center h-64 gap-4">
      <div className="text-[var(--red)]">{message}</div>
      <button
        onClick={onRetry}
        className="px-4 py-2 text-sm bg-[var(--bg-surface)] border border-[var(--border)] rounded hover:bg-[var(--bg-elevated)]"
      >
        Retry
      </button>
    </div>
  )
}
