import { clsx } from 'clsx'
import { useEffect, useRef, useState } from 'react'

interface StatCardProps {
  label: string
  value: string | number
  subValue?: string
  trend?: 'up' | 'down' | 'neutral'
  large?: boolean
}

export function StatCard({ label, value, subValue, trend, large }: StatCardProps) {
  const [flash, setFlash] = useState<'up' | 'down' | null>(null)
  const prevValue = useRef(value)

  useEffect(() => {
    if (typeof value === 'number' && typeof prevValue.current === 'number') {
      if (value > prevValue.current) {
        setFlash('up')
      } else if (value < prevValue.current) {
        setFlash('down')
      }
      const timer = setTimeout(() => setFlash(null), 500)
      prevValue.current = value
      return () => clearTimeout(timer)
    }
    prevValue.current = value
  }, [value])

  return (
    <div
      className={clsx(
        'rounded border border-[var(--border)] bg-[var(--bg-surface)] p-4',
        flash === 'up' && 'flash-up',
        flash === 'down' && 'flash-down'
      )}
    >
      <div className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-1">
        {label}
      </div>
      <div
        className={clsx(
          'font-mono font-semibold',
          large ? 'text-3xl' : 'text-xl',
          trend === 'up' && 'text-[var(--green)]',
          trend === 'down' && 'text-[var(--red)]',
          trend === 'neutral' && 'text-[var(--text-primary)]'
        )}
      >
        {typeof value === 'number' ? formatNumber(value) : value}
      </div>
      {subValue && (
        <div className="text-xs text-[var(--text-secondary)] mt-1 font-mono">
          {subValue}
        </div>
      )}
    </div>
  )
}

function formatNumber(n: number): string {
  if (Math.abs(n) >= 1000000) {
    return (n / 1000000).toFixed(2) + 'M'
  }
  if (Math.abs(n) >= 1000) {
    return (n / 1000).toFixed(2) + 'K'
  }
  return n.toFixed(2)
}
