import { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { DataTable, type Column } from '@/components'
import { getXGBoostFeatures, getXGBoostMetrics } from '@/api/client'
import type { FeatureImportance, XGBoostMetric } from '@/types'

interface WalkForwardFold {
  fold: number
  train_start: string
  train_end: string
  val_start: string
  val_end: string
  ic: number
  brier: number
  accuracy: number
}

export function XGBoostModel() {
  const [features, setFeatures] = useState<FeatureImportance[]>([])
  const [metrics, setMetrics] = useState<XGBoostMetric[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    try {
      const [featData, metData] = await Promise.all([
        getXGBoostFeatures(),
        getXGBoostMetrics(),
      ])
      setFeatures(featData)
      setMetrics(metData)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch XGBoost data')
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

  // Sort features by importance and take top 10
  const topFeatures = [...features]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10)

  // Mock walk-forward folds (would come from backend in production)
  const mockFolds: WalkForwardFold[] = [
    { fold: 1, train_start: '2024-01-01', train_end: '2024-03-31', val_start: '2024-04-01', val_end: '2024-04-30', ic: 0.082, brier: 0.21, accuracy: 0.58 },
    { fold: 2, train_start: '2024-02-01', train_end: '2024-04-30', val_start: '2024-05-01', val_end: '2024-05-31', ic: 0.071, brier: 0.23, accuracy: 0.55 },
    { fold: 3, train_start: '2024-03-01', train_end: '2024-05-31', val_start: '2024-06-01', val_end: '2024-06-30', ic: 0.068, brier: 0.22, accuracy: 0.56 },
    { fold: 4, train_start: '2024-04-01', train_end: '2024-06-30', val_start: '2024-07-01', val_end: '2024-07-31', ic: 0.079, brier: 0.20, accuracy: 0.59 },
    { fold: 5, train_start: '2024-05-01', train_end: '2024-07-31', val_start: '2024-08-01', val_end: '2024-08-31', ic: 0.085, brier: 0.19, accuracy: 0.61 },
  ]

  const foldColumns: Column<WalkForwardFold>[] = [
    { key: 'fold', header: 'Fold', mono: true },
    { key: 'train_start', header: 'Train Start', mono: true },
    { key: 'train_end', header: 'Train End', mono: true },
    { key: 'val_start', header: 'Val Start', mono: true },
    { key: 'val_end', header: 'Val End', mono: true },
    {
      key: 'ic',
      header: 'IC',
      align: 'right',
      mono: true,
      render: (row) => (
        <span className={row.ic >= 0.05 ? 'text-[var(--green)]' : 'text-[var(--text-secondary)]'}>
          {row.ic.toFixed(3)}
        </span>
      ),
    },
    {
      key: 'brier',
      header: 'Brier',
      align: 'right',
      mono: true,
      render: (row) => (
        <span className={row.brier <= 0.25 ? 'text-[var(--green)]' : 'text-[var(--amber)]'}>
          {row.brier.toFixed(3)}
        </span>
      ),
    },
    {
      key: 'accuracy',
      header: 'Acc',
      align: 'right',
      mono: true,
      render: (row) => `${(row.accuracy * 100).toFixed(1)}%`,
    },
  ]

  const latestMetrics = metrics.length > 0 ? metrics[metrics.length - 1] : null

  return (
    <div className="p-6 space-y-6">
      {/* Model metadata card */}
      <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
        <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
          Current Model Configuration
        </h3>
        <div className="grid grid-cols-6 gap-4">
          <ModelStat label="n_estimators" value="200" />
          <ModelStat label="max_depth" value="6" />
          <ModelStat label="learning_rate" value="0.05" />
          <ModelStat label="num_features" value={features.length.toString()} />
          <ModelStat label="Last Retrain" value={latestMetrics?.timestamp ? formatDate(latestMetrics.timestamp) : '--'} />
          <ModelStat
            label="Latest IC"
            value={latestMetrics?.ic?.toFixed(3) ?? '--'}
            highlight={latestMetrics !== null && latestMetrics.ic >= 0.05}
          />
        </div>
      </div>

      {/* SHAP Feature Importance */}
      <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
        <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
          SHAP Feature Importance (Top 10)
        </h3>
        <div style={{ height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={topFeatures} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis type="number" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <YAxis
                type="category"
                dataKey="feature"
                stroke="var(--text-muted)"
                tick={{ fontSize: 11 }}
                width={150}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="importance" fill="var(--accent)" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* IC and Brier trends side by side */}
      <div className="grid grid-cols-2 gap-4">
        {/* IC Over Time */}
        <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
          <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
            Information Coefficient Over Time
          </h3>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatDate}
                  stroke="var(--text-muted)"
                  tick={{ fontSize: 11 }}
                />
                <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} domain={[0, 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine
                  y={0.05}
                  stroke="var(--green)"
                  strokeDasharray="5 5"
                  label={{ value: 'Threshold (0.05)', fill: 'var(--green)', fontSize: 10, position: 'right' }}
                />
                <Line
                  type="monotone"
                  dataKey="ic"
                  stroke="var(--accent)"
                  strokeWidth={2}
                  dot={{ fill: 'var(--accent)', strokeWidth: 0, r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Brier Score Trend */}
        <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
          <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
            Brier Score Trend (Lower is Better)
          </h3>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatDate}
                  stroke="var(--text-muted)"
                  tick={{ fontSize: 11 }}
                />
                <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} domain={[0, 0.5]} reversed />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine
                  y={0.25}
                  stroke="var(--amber)"
                  strokeDasharray="5 5"
                  label={{ value: 'Threshold (0.25)', fill: 'var(--amber)', fontSize: 10, position: 'right' }}
                />
                <Line
                  type="monotone"
                  dataKey="brier"
                  stroke="var(--red)"
                  strokeWidth={2}
                  dot={{ fill: 'var(--red)', strokeWidth: 0, r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Walk-Forward Folds Table */}
      <div>
        <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-3">
          Walk-Forward Cross-Validation Folds
        </h3>
        <DataTable
          columns={foldColumns}
          data={mockFolds}
          keyField="fold"
          emptyMessage="No fold data available"
        />
      </div>
    </div>
  )
}

function ModelStat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-1">
        {label}
      </div>
      <div className={`font-mono text-lg ${highlight ? 'text-[var(--green)]' : 'text-[var(--text-primary)]'}`}>
        {value}
      </div>
    </div>
  )
}

interface TooltipProps {
  active?: boolean
  payload?: Array<{ value: number; dataKey: string }>
  label?: string
}

function CustomTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null

  return (
    <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded px-3 py-2 text-sm">
      <div className="text-[var(--text-muted)] text-xs mb-1">{label}</div>
      {payload.map((entry, i) => (
        <div key={i} className="font-mono text-[var(--text-primary)]">
          {entry.dataKey}: {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
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

function LoadingState() {
  return (
    <div className="p-6 flex items-center justify-center h-64">
      <div className="text-[var(--text-muted)]">Loading XGBoost data...</div>
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
