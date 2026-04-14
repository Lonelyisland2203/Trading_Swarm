import { useEffect, useState } from 'react'
import { clsx } from 'clsx'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { AlertTriangle } from 'lucide-react'
import { getFundingRates, getOpenInterest } from '@/api/client'
import type { FundingRates, OpenInterest } from '@/types'

interface LiquidationEvent {
  timestamp: string
  symbol: string
  side: 'LONG' | 'SHORT'
  size_usd: number
}

export function RiskMonitor() {
  const [funding, setFunding] = useState<FundingRates>({})
  const [oi, setOi] = useState<OpenInterest>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Mock liquidation events - in production would come from WebSocket feed
  const [liquidations] = useState<LiquidationEvent[]>([
    { timestamp: '2024-12-15T14:32:00Z', symbol: 'BTC', side: 'LONG', size_usd: 2500000 },
    { timestamp: '2024-12-15T14:28:00Z', symbol: 'ETH', side: 'SHORT', size_usd: 1200000 },
    { timestamp: '2024-12-15T14:15:00Z', symbol: 'SOL', side: 'LONG', size_usd: 800000 },
    { timestamp: '2024-12-15T13:55:00Z', symbol: 'BTC', side: 'LONG', size_usd: 3100000 },
    { timestamp: '2024-12-15T13:42:00Z', symbol: 'ETH', side: 'LONG', size_usd: 950000 },
  ])

  const fetchData = async () => {
    try {
      const [fundingData, oiData] = await Promise.all([
        getFundingRates(),
        getOpenInterest(),
      ])
      setFunding(fundingData)
      setOi(oiData)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch risk data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 60000) // Poll every 60s for risk data
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return <LoadingState />
  }

  if (error) {
    return <ErrorState message={error} onRetry={fetchData} />
  }

  // Prepare funding heatmap data
  const fundingData = Object.entries(funding).map(([symbol, rate]) => ({
    symbol,
    rate,
    annualized: rate * 3 * 365 * 100, // Convert to annualized percentage
  }))

  // Prepare OI change data (mock 24h change - would need historical data in production)
  const oiData = Object.entries(oi).map(([symbol, data]) => ({
    symbol,
    oi: data.oi,
    change24h: Math.random() * 20 - 10, // Mock: -10% to +10%
  }))

  return (
    <div className="p-6 space-y-6">
      {/* Funding Rate Heatmap */}
      <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
        <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
          Funding Rates (8h)
        </h3>
        <div className="grid grid-cols-6 gap-2">
          {fundingData.length > 0 ? (
            fundingData.map((item) => (
              <FundingCell key={item.symbol} symbol={item.symbol} rate={item.rate} />
            ))
          ) : (
            <div className="col-span-6 text-center text-[var(--text-muted)] py-8">
              No funding rate data available
            </div>
          )}
        </div>
        <div className="flex items-center justify-center gap-4 mt-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded bg-[var(--green)]" />
            <span className="text-[var(--text-muted)]">Negative (pay shorts)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded bg-[var(--border)]" />
            <span className="text-[var(--text-muted)]">Neutral</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded bg-[var(--red)]" />
            <span className="text-[var(--text-muted)]">Positive (pay longs)</span>
          </div>
        </div>
      </div>

      {/* OI Changes */}
      <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
        <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
          Open Interest (24h Change)
        </h3>
        <div style={{ height: 200 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={oiData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis
                type="number"
                stroke="var(--text-muted)"
                tick={{ fontSize: 11 }}
                tickFormatter={(v) => `${v.toFixed(1)}%`}
              />
              <YAxis
                type="category"
                dataKey="symbol"
                stroke="var(--text-muted)"
                tick={{ fontSize: 11 }}
                width={60}
              />
              <Tooltip content={<OiTooltip />} />
              <Bar dataKey="change24h" radius={[0, 4, 4, 0]}>
                {oiData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.change24h >= 0 ? 'var(--green)' : 'var(--red)'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* OI Current Values */}
      <div className="grid grid-cols-3 gap-4">
        {oiData.map((item) => (
          <div
            key={item.symbol}
            className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4"
          >
            <div className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-1">
              {item.symbol} Open Interest
            </div>
            <div className="font-mono text-xl text-[var(--text-primary)]">
              ${formatLargeNumber(item.oi)}
            </div>
            <div
              className={clsx(
                'text-xs font-mono mt-1',
                item.change24h >= 0 ? 'text-[var(--green)]' : 'text-[var(--red)]'
              )}
            >
              {item.change24h >= 0 ? '+' : ''}{item.change24h.toFixed(2)}% (24h)
            </div>
          </div>
        ))}
      </div>

      {/* Liquidation Alerts */}
      <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="w-4 h-4 text-[var(--amber)]" />
          <h3 className="text-sm font-medium text-[var(--text-secondary)]">
            Large Liquidations (Recent)
          </h3>
        </div>
        <div className="space-y-2 max-h-[300px] overflow-auto">
          {liquidations.length > 0 ? (
            liquidations.map((liq, idx) => (
              <div
                key={idx}
                className={clsx(
                  'flex items-center justify-between py-2 px-3 rounded',
                  liq.side === 'LONG' ? 'bg-[var(--red-dim)]' : 'bg-[var(--green-dim)]'
                )}
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm font-medium">{liq.symbol}</span>
                  <span
                    className={clsx(
                      'px-2 py-0.5 text-xs font-medium rounded',
                      liq.side === 'LONG'
                        ? 'bg-[var(--red)] text-white'
                        : 'bg-[var(--green)] text-[var(--bg-primary)]'
                    )}
                  >
                    {liq.side}
                  </span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="font-mono text-sm text-[var(--text-primary)]">
                    ${formatLargeNumber(liq.size_usd)}
                  </span>
                  <span className="text-xs text-[var(--text-muted)]">
                    {formatTime(liq.timestamp)}
                  </span>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center text-[var(--text-muted)] py-8">
              No recent large liquidations
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function FundingCell({ symbol, rate }: { symbol: string; rate: number }) {
  // Scale rate for visual effect (-0.01 to 0.01 typical range)
  const intensity = Math.min(Math.abs(rate) * 100, 1)
  const isPositive = rate > 0

  const bgColor = rate === 0
    ? 'var(--border)'
    : isPositive
      ? `rgba(226, 75, 74, ${0.2 + intensity * 0.6})`
      : `rgba(0, 208, 132, ${0.2 + intensity * 0.6})`

  return (
    <div
      className="rounded p-3 text-center"
      style={{ backgroundColor: bgColor }}
    >
      <div className="text-xs font-medium text-[var(--text-primary)]">{symbol}</div>
      <div
        className={clsx(
          'font-mono text-sm mt-1',
          isPositive ? 'text-[var(--red)]' : 'text-[var(--green)]'
        )}
      >
        {(rate * 100).toFixed(4)}%
      </div>
      <div className="text-xs text-[var(--text-muted)]">
        {(rate * 3 * 365 * 100).toFixed(1)}% APR
      </div>
    </div>
  )
}

interface OiTooltipProps {
  active?: boolean
  payload?: Array<{ payload: { symbol: string; oi: number; change24h: number } }>
}

function OiTooltip({ active, payload }: OiTooltipProps) {
  if (!active || !payload || payload.length === 0) return null

  const data = payload[0].payload

  return (
    <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded px-3 py-2 text-sm">
      <div className="font-medium text-[var(--text-primary)]">{data.symbol}</div>
      <div className="font-mono text-[var(--text-secondary)]">
        OI: ${formatLargeNumber(data.oi)}
      </div>
      <div
        className={clsx(
          'font-mono',
          data.change24h >= 0 ? 'text-[var(--green)]' : 'text-[var(--red)]'
        )}
      >
        {data.change24h >= 0 ? '+' : ''}{data.change24h.toFixed(2)}%
      </div>
    </div>
  )
}

function formatLargeNumber(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(2) + 'K'
  return n.toFixed(2)
}

function formatTime(ts: string): string {
  const date = new Date(ts)
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}

function LoadingState() {
  return (
    <div className="p-6 flex items-center justify-center h-64">
      <div className="text-[var(--text-muted)]">Loading risk data...</div>
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
