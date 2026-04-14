import { useEffect, useState } from 'react'
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Scatter,
  ScatterChart,
  ZAxis,
} from 'recharts'
import { clsx } from 'clsx'
import { DataTable, type Column } from '@/components'
import { getAutoresearchLog } from '@/api/client'
import type { AutoresearchExperiment } from '@/types'

export function Autoresearch() {
  const [experiments, setExperiments] = useState<AutoresearchExperiment[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    try {
      const data = await getAutoresearchLog()
      setExperiments(data)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch autoresearch log')
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

  // Add experiment numbers for charting
  const chartData = experiments.map((exp, idx) => ({
    ...exp,
    experimentNum: idx + 1,
    kept: exp.kept_or_reverted === 'kept',
  }))

  const keptExperiments = chartData.filter((e) => e.kept)
  const revertedExperiments = chartData.filter((e) => !e.kept)

  // Find best config (latest kept experiment)
  const bestConfig = keptExperiments.length > 0
    ? keptExperiments[keptExperiments.length - 1]
    : null

  const columns: Column<AutoresearchExperiment & { experimentNum: number }>[] = [
    { key: 'experimentNum', header: '#', mono: true, align: 'center' },
    {
      key: 'timestamp',
      header: 'Timestamp',
      mono: true,
      render: (row) => formatTimestamp(row.timestamp),
    },
    {
      key: 'change_description',
      header: 'Change',
      render: (row) => (
        <span className="text-[var(--text-secondary)] text-xs truncate max-w-[300px] block">
          {row.change_description}
        </span>
      ),
    },
    {
      key: 'sharpe_net',
      header: 'Sharpe_net',
      align: 'right',
      mono: true,
      render: (row) => (
        <span className={row.sharpe_net >= 0.02 ? 'text-[var(--green)]' : 'text-[var(--text-secondary)]'}>
          {row.sharpe_net?.toFixed(4) ?? '--'}
        </span>
      ),
    },
    {
      key: 'ic',
      header: 'IC',
      align: 'right',
      mono: true,
      render: (row) => row.ic?.toFixed(3) ?? '--',
    },
    {
      key: 'brier',
      header: 'Brier',
      align: 'right',
      mono: true,
      render: (row) => row.brier?.toFixed(3) ?? '--',
    },
    {
      key: 'accuracy',
      header: 'Acc',
      align: 'right',
      mono: true,
      render: (row) => row.accuracy ? `${(row.accuracy * 100).toFixed(1)}%` : '--',
    },
    {
      key: 'kept_or_reverted',
      header: 'Status',
      align: 'center',
      render: (row) => (
        <span
          className={clsx(
            'px-2 py-0.5 text-xs font-medium rounded',
            row.kept_or_reverted === 'kept'
              ? 'bg-[var(--green)] text-[var(--bg-primary)]'
              : 'bg-[var(--red)] text-white'
          )}
        >
          {row.kept_or_reverted.toUpperCase()}
        </span>
      ),
    },
  ]

  return (
    <div className="p-6 space-y-6">
      {/* Best config card */}
      {bestConfig && (
        <div className="rounded border border-[var(--green)] bg-[var(--green-dim)] p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-[var(--green)]">
              Current Best Configuration
            </h3>
            <span className="text-xs text-[var(--text-muted)]">
              Experiment #{bestConfig.experimentNum}
            </span>
          </div>
          <div className="grid grid-cols-5 gap-4 text-sm">
            <div>
              <div className="text-xs text-[var(--text-muted)]">Sharpe_net</div>
              <div className="font-mono text-lg text-[var(--green)]">
                {bestConfig.sharpe_net?.toFixed(4) ?? '--'}
              </div>
            </div>
            <div>
              <div className="text-xs text-[var(--text-muted)]">IC</div>
              <div className="font-mono text-lg">{bestConfig.ic?.toFixed(3) ?? '--'}</div>
            </div>
            <div>
              <div className="text-xs text-[var(--text-muted)]">Brier</div>
              <div className="font-mono text-lg">{bestConfig.brier?.toFixed(3) ?? '--'}</div>
            </div>
            <div>
              <div className="text-xs text-[var(--text-muted)]">Accuracy</div>
              <div className="font-mono text-lg">
                {bestConfig.accuracy ? `${(bestConfig.accuracy * 100).toFixed(1)}%` : '--'}
              </div>
            </div>
            <div>
              <div className="text-xs text-[var(--text-muted)]">Date</div>
              <div className="font-mono text-lg">{formatTimestamp(bestConfig.timestamp)}</div>
            </div>
          </div>
          <div className="mt-3 text-xs text-[var(--text-secondary)]">
            {bestConfig.change_description}
          </div>
        </div>
      )}

      {/* Running experiment indicator */}
      <div className="flex items-center gap-2">
        <div className="relative">
          <div className="w-2 h-2 rounded-full bg-[var(--amber)]" />
          <div className="absolute inset-0 w-2 h-2 rounded-full bg-[var(--amber)] animate-ping opacity-75" />
        </div>
        <span className="text-xs text-[var(--text-muted)]">
          Autoresearch may be running - check process status
        </span>
      </div>

      {/* Sharpe progression scatter plot */}
      <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4">
        <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
          Sharpe_net Progression by Experiment
        </h3>
        <div style={{ height: 250 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis
                type="number"
                dataKey="experimentNum"
                name="Experiment"
                stroke="var(--text-muted)"
                tick={{ fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="sharpe_net"
                name="Sharpe_net"
                stroke="var(--text-muted)"
                tick={{ fontSize: 11 }}
                domain={['auto', 'auto']}
              />
              <ZAxis range={[60, 60]} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine
                y={0.02}
                stroke="var(--amber)"
                strokeDasharray="5 5"
                label={{ value: 'Threshold (0.02)', fill: 'var(--amber)', fontSize: 10, position: 'right' }}
              />
              <Scatter
                name="Kept"
                data={keptExperiments}
                fill="var(--green)"
              />
              <Scatter
                name="Reverted"
                data={revertedExperiments}
                fill="var(--red)"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="flex items-center gap-4 mt-2 justify-center text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-[var(--green)]" />
            <span className="text-[var(--text-muted)]">Kept</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-[var(--red)]" />
            <span className="text-[var(--text-muted)]">Reverted</span>
          </div>
        </div>
      </div>

      {/* Experiment log table */}
      <div>
        <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-3">
          Experiment Log ({experiments.length} experiments)
        </h3>
        <DataTable
          columns={columns}
          data={chartData}
          keyField="experiment_id"
          emptyMessage="No experiments recorded"
          maxHeight="400px"
          rowClassName={(row) =>
            row.kept_or_reverted === 'kept' ? 'bg-[var(--green-dim)]' : ''
          }
        />
      </div>
    </div>
  )
}

interface TooltipProps {
  active?: boolean
  payload?: Array<{ payload: AutoresearchExperiment & { experimentNum: number } }>
}

function CustomTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null

  const data = payload[0].payload

  return (
    <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded px-3 py-2 text-sm max-w-[300px]">
      <div className="text-[var(--text-muted)] text-xs mb-1">
        Experiment #{data.experimentNum}
      </div>
      <div className="font-mono text-[var(--text-primary)]">
        Sharpe_net: {data.sharpe_net?.toFixed(4) ?? '--'}
      </div>
      <div className="text-xs text-[var(--text-secondary)] mt-1 truncate">
        {data.change_description}
      </div>
      <div className={clsx(
        'text-xs mt-1',
        data.kept_or_reverted === 'kept' ? 'text-[var(--green)]' : 'text-[var(--red)]'
      )}>
        {data.kept_or_reverted.toUpperCase()}
      </div>
    </div>
  )
}

function formatTimestamp(ts: string): string {
  if (!ts) return '--'
  const date = new Date(ts)
  return date.toLocaleDateString('en-US', {
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
      <div className="text-[var(--text-muted)]">Loading autoresearch log...</div>
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
