import { clsx } from 'clsx'

interface StatusBadgeProps {
  status: 'running' | 'stopped' | 'error' | 'active' | 'inactive' | 'warning'
  label: string
  pulse?: boolean
}

export function StatusBadge({ status, label, pulse }: StatusBadgeProps) {
  const colorMap = {
    running: 'bg-[var(--green)]',
    active: 'bg-[var(--green)]',
    stopped: 'bg-[var(--text-muted)]',
    inactive: 'bg-[var(--text-muted)]',
    warning: 'bg-[var(--amber)]',
    error: 'bg-[var(--red)]',
  }

  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div
          className={clsx(
            'w-2 h-2 rounded-full',
            colorMap[status]
          )}
        />
        {pulse && (status === 'running' || status === 'active') && (
          <div
            className={clsx(
              'absolute inset-0 w-2 h-2 rounded-full animate-ping',
              colorMap[status],
              'opacity-75'
            )}
          />
        )}
      </div>
      <span className="text-sm text-[var(--text-secondary)]">{label}</span>
    </div>
  )
}
