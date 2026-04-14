import { useEffect, useState } from 'react'
import { Activity, AlertTriangle } from 'lucide-react'
import { clsx } from 'clsx'
import { StatCard, StatusBadge, DrawdownGauge, DataTable, type Column } from '@/components'
import { useWebSocket } from '@/hooks/useWebSocket'
import type { Position } from '@/types'

export function LiveOverview() {
  const { data, isConnected, error, reconnect } = useWebSocket()
  const [lastUpdate, setLastUpdate] = useState<string>('')

  useEffect(() => {
    if (data?.timestamp) {
      setLastUpdate(new Date(data.timestamp).toLocaleTimeString())
    }
  }, [data?.timestamp])

  const positions = data?.positions ?? []
  const dailyPnl = data?.daily_pnl ?? []
  const health = data?.health ?? {}

  const totalPnL = positions.reduce((sum, p) => sum + p.unrealized_pnl, 0)
  const totalDailyPnL = dailyPnl.reduce((sum, d) => sum + d.pnl, 0)
  const dailyDrawdown = health.daily_loss_pct ?? 0
  const isKillSwitchActive = health.kill_switch_active ?? false

  const watchdogAge = getWatchdogAge(health.watchdog_heartbeat)
  const signalLoopStatus = health.signal_loop_status ?? 'stopped'

  const positionColumns: Column<Position>[] = [
    { key: 'symbol', header: 'Symbol', mono: true },
    {
      key: 'side',
      header: 'Side',
      render: (row) => (
        <span className={row.side === 'LONG' ? 'text-[var(--green)]' : 'text-[var(--red)]'}>
          {row.side}
        </span>
      ),
    },
    { key: 'amount', header: 'Size', align: 'right', mono: true, render: (row) => row.amount.toFixed(4) },
    { key: 'entry_price', header: 'Entry', align: 'right', mono: true, render: (row) => formatPrice(row.entry_price) },
    { key: 'mark_price', header: 'Mark', align: 'right', mono: true, render: (row) => formatPrice(row.mark_price) },
    {
      key: 'unrealized_pnl',
      header: 'Unreal. P&L',
      align: 'right',
      mono: true,
      render: (row) => (
        <span className={row.unrealized_pnl >= 0 ? 'text-[var(--green)]' : 'text-[var(--red)]'}>
          {row.unrealized_pnl >= 0 ? '+' : ''}{row.unrealized_pnl.toFixed(2)}
        </span>
      ),
    },
    {
      key: 'duration',
      header: 'Duration',
      align: 'right',
      render: () => <span className="text-[var(--text-muted)]">--</span>, // Placeholder - needs actual duration tracking
    },
  ]

  return (
    <div className="p-6 space-y-6">
      {/* Connection status bar */}
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-4">
          <StatusBadge
            status={isConnected ? 'running' : 'error'}
            label={isConnected ? 'Connected' : 'Disconnected'}
            pulse={isConnected}
          />
          {lastUpdate && (
            <span className="text-xs text-[var(--text-muted)]">
              Last update: {lastUpdate}
            </span>
          )}
        </div>
        {error && (
          <button
            onClick={reconnect}
            className="text-xs text-[var(--accent)] hover:underline"
          >
            Reconnect
          </button>
        )}
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Total P&L"
          value={totalPnL}
          trend={totalPnL >= 0 ? 'up' : 'down'}
          large
          subValue={`Daily: ${totalDailyPnL >= 0 ? '+' : ''}${totalDailyPnL.toFixed(2)}`}
        />

        <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4 flex flex-col items-center justify-center relative">
          <div className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2">
            Daily Drawdown
          </div>
          <DrawdownGauge current={Math.abs(dailyDrawdown)} max={2} size={100} />
        </div>

        <StatCard
          label="Open Positions"
          value={positions.length}
          trend="neutral"
          subValue={positions.length > 0 ? summarizePositions(positions) : 'None'}
        />

        <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4 space-y-3">
          <div className="text-xs uppercase tracking-wider text-[var(--text-muted)]">
            System Status
          </div>
          <div className="space-y-2">
            <StatusBadge
              status={isKillSwitchActive ? 'error' : 'active'}
              label={isKillSwitchActive ? 'Kill Switch ACTIVE' : 'Kill Switch Off'}
            />
            <StatusBadge
              status={watchdogAge > 60 ? 'error' : 'running'}
              label={`Watchdog ${watchdogAge > 0 ? `${watchdogAge}s ago` : 'OK'}`}
              pulse={watchdogAge <= 60}
            />
            <StatusBadge
              status={signalLoopStatus === 'running' ? 'running' : signalLoopStatus === 'error' ? 'error' : 'stopped'}
              label={`Signal Loop: ${signalLoopStatus}`}
            />
          </div>
        </div>
      </div>

      {/* Positions table */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Activity className="w-4 h-4 text-[var(--text-muted)]" />
          <h2 className="text-sm font-medium text-[var(--text-secondary)]">
            Open Positions
          </h2>
        </div>
        <DataTable
          columns={positionColumns}
          data={positions}
          keyField="symbol"
          emptyMessage="No open positions"
          rowClassName={(row) =>
            clsx(
              row.unrealized_pnl >= 0
                ? 'bg-[var(--green-dim)]'
                : 'bg-[var(--red-dim)]'
            )
          }
        />
      </div>

      {/* Alerts section */}
      {(isKillSwitchActive || watchdogAge > 60) && (
        <div className="rounded border border-[var(--red)] bg-[var(--red-dim)] p-4">
          <div className="flex items-center gap-2 text-[var(--red)]">
            <AlertTriangle className="w-4 h-4" />
            <span className="font-medium">System Alert</span>
          </div>
          <ul className="mt-2 space-y-1 text-sm text-[var(--text-secondary)]">
            {isKillSwitchActive && (
              <li>Kill switch is active - all trading halted</li>
            )}
            {watchdogAge > 60 && (
              <li>Watchdog heartbeat stale ({watchdogAge}s ago) - check watchdog process</li>
            )}
          </ul>
        </div>
      )}
    </div>
  )
}

function formatPrice(price: number): string {
  if (price >= 1000) return price.toFixed(2)
  if (price >= 1) return price.toFixed(4)
  return price.toFixed(6)
}

function summarizePositions(positions: Position[]): string {
  const longs = positions.filter((p) => p.side === 'LONG').length
  const shorts = positions.filter((p) => p.side === 'SHORT').length
  return `${longs}L / ${shorts}S`
}

function getWatchdogAge(heartbeat?: string): number {
  if (!heartbeat) return -1
  const diff = Date.now() - new Date(heartbeat).getTime()
  return Math.floor(diff / 1000)
}
