import { useEffect, useState } from 'react'
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  BarChart,
  Bar,
  Cell,
} from 'recharts'
import { StatCard } from '@/components'
import { getPerformance, getDailyPnL } from '@/api/client'
import type { PerformanceData, DailyPnL } from '@/types'

export function Performance() {
  const [data, setData] = useState<PerformanceData | null>(null)
  const [dailyPnL, setDailyPnL] = useState<DailyPnL[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    try {
      const [perfData, pnlData] = await Promise.all([
        getPerformance(),
        getDailyPnL(),
      ])
      setData(perfData)
      setDailyPnL(pnlData)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch performance data')
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

  if (error || !data) {
    return <ErrorState message={error ?? 'No data'} onRetry={fetchData} />
  }

  const totalReturn = calculateTotalReturn(data.equity_curve)
  const avgWinLossRatio = calculateAvgWinLossRatio(dailyPnL)
  const totalFees = calculateTotalFees(dailyPnL)
  const latestSharpe = data.rolling_sharpe.length > 0
    ? data.rolling_sharpe[data.rolling_sharpe.length - 1].sharpe
    : 0

  return (
    <div className="p-6 space-y-6">
      {/* Stats row */}
      <div className="grid grid-cols-5 gap-4">
        <StatCard
          label="Total Return"
          value={`${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`}
          trend={totalReturn >= 0 ? 'up' : 'down'}
        />
        <StatCard
          label="Win Rate"
          value={`${(data.win_rate * 100).toFixed(1)}%`}
          trend={data.win_rate >= 0.5 ? 'up' : 'down'}
        />
        <StatCard
          label="Sharpe (30d)"
          value={latestSharpe.toFixed(2)}
          trend={latestSharpe >= 1 ? 'up' : latestSharpe >= 0 ? 'neutral' : 'down'}
        />
        <StatCard
          label="Avg Win/Loss"
          value={avgWinLossRatio.toFixed(2)}
          trend={avgWinLossRatio >= 1.5 ? 'up' : 'neutral'}
        />
        <StatCard
          label="Total Fees"
          value={`-${totalFees.toFixed(2)}`}
          trend="down"
        />
      </div>

      {/* Equity curve */}
      <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
        <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
          Equity Curve
        </h3>
        <div style={{ height: 250 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data.equity_curve}>
              <defs>
                <linearGradient id="equityGradientPos" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--green)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="var(--green)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatDate}
                stroke="var(--text-muted)"
                tick={{ fontSize: 11 }}
              />
              <YAxis
                stroke="var(--text-muted)"
                tick={{ fontSize: 11 }}
                tickFormatter={(v) => `$${v.toLocaleString()}`}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={10000} stroke="var(--chart-line)" strokeDasharray="5 5" />
              <Area
                type="monotone"
                dataKey="equity"
                stroke="var(--green)"
                fill="url(#equityGradientPos)"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Rolling Sharpe + Drawdown side by side */}
      <div className="grid grid-cols-2 gap-4">
        {/* Rolling Sharpe */}
        <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
          <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
            Rolling 30-Day Sharpe Ratio
          </h3>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.rolling_sharpe}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatDate}
                  stroke="var(--text-muted)"
                  tick={{ fontSize: 11 }}
                />
                <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0} stroke="var(--chart-line)" strokeDasharray="5 5" />
                <ReferenceLine
                  y={1}
                  stroke="var(--green)"
                  strokeDasharray="5 5"
                  label={{ value: 'Target', fill: 'var(--green)', fontSize: 10 }}
                />
                <Line
                  type="monotone"
                  dataKey="sharpe"
                  stroke="var(--accent)"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Drawdown */}
        <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
          <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
            Drawdown
          </h3>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data.drawdown}>
                <defs>
                  <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--red)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="var(--red)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatDate}
                  stroke="var(--text-muted)"
                  tick={{ fontSize: 11 }}
                />
                <YAxis
                  stroke="var(--text-muted)"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v) => `${v.toFixed(0)}%`}
                  reversed
                />
                <Tooltip content={<CustomTooltip unit="%" />} />
                <Area
                  type="monotone"
                  dataKey="drawdown"
                  stroke="var(--red)"
                  fill="url(#drawdownGradient)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Fee drag waterfall */}
      <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
        <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
          P&L Breakdown (Fee Drag)
        </h3>
        <div style={{ height: 200 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={getWaterfallData(dailyPnL)}
              layout="vertical"
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis type="number" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <YAxis
                type="category"
                dataKey="name"
                stroke="var(--text-muted)"
                tick={{ fontSize: 11 }}
                width={100}
              />
              <Tooltip content={<CustomTooltip prefix="$" />} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {getWaterfallData(dailyPnL).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

interface TooltipProps {
  active?: boolean
  payload?: Array<{ value: number; dataKey: string }>
  label?: string
  unit?: string
  prefix?: string
}

function CustomTooltip({ active, payload, label, unit = '', prefix = '' }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null

  return (
    <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded px-3 py-2 text-sm">
      <div className="text-[var(--text-muted)] text-xs mb-1">{formatDate(label ?? '')}</div>
      {payload.map((entry, i) => (
        <div key={i} className="font-mono text-[var(--text-primary)]">
          {prefix}{typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}{unit}
        </div>
      ))}
    </div>
  )
}

function formatDate(ts: string): string {
  if (!ts) return ''
  const date = new Date(ts)
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function calculateTotalReturn(equityCurve: PerformanceData['equity_curve']): number {
  if (equityCurve.length < 2) return 0
  const initial = equityCurve[0].equity
  const final = equityCurve[equityCurve.length - 1].equity
  return ((final - initial) / initial) * 100
}

function calculateAvgWinLossRatio(dailyPnL: DailyPnL[]): number {
  const wins = dailyPnL.filter((d) => d.pnl > 0)
  const losses = dailyPnL.filter((d) => d.pnl < 0)
  if (wins.length === 0 || losses.length === 0) return 0
  const avgWin = wins.reduce((sum, d) => sum + d.pnl, 0) / wins.length
  const avgLoss = Math.abs(losses.reduce((sum, d) => sum + d.pnl, 0) / losses.length)
  return avgLoss > 0 ? avgWin / avgLoss : 0
}

function calculateTotalFees(_dailyPnL: DailyPnL[]): number {
  // Placeholder - actual fee tracking would need to come from backend
  return 0
}

function getWaterfallData(dailyPnL: DailyPnL[]) {
  const grossPnL = dailyPnL.reduce((sum, d) => sum + d.pnl, 0)
  const fees = grossPnL * 0.001 // Estimate 0.1% fees
  const slippage = grossPnL * 0.0005 // Estimate 0.05% slippage
  const netPnL = grossPnL - fees - slippage

  return [
    { name: 'Gross P&L', value: grossPnL, color: grossPnL >= 0 ? 'var(--green)' : 'var(--red)' },
    { name: 'Fees', value: -fees, color: 'var(--amber)' },
    { name: 'Slippage', value: -slippage, color: 'var(--amber)' },
    { name: 'Net P&L', value: netPnL, color: netPnL >= 0 ? 'var(--green)' : 'var(--red)' },
  ]
}

function LoadingState() {
  return (
    <div className="p-6 flex items-center justify-center h-64">
      <div className="text-[var(--text-muted)]">Loading performance data...</div>
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
