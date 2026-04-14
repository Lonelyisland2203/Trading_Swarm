import { clsx } from 'clsx'

interface DrawdownGaugeProps {
  current: number  // Current drawdown as percentage (0-100)
  max: number      // Max allowed drawdown (default 2%)
  size?: number    // Size in pixels
}

export function DrawdownGauge({ current, max = 2, size = 120 }: DrawdownGaugeProps) {
  const percentage = Math.min((current / max) * 100, 100)
  const radius = (size - 12) / 2
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (percentage / 100) * circumference

  // Color based on percentage
  const getColor = () => {
    if (percentage < 50) return 'var(--green)'
    if (percentage < 80) return 'var(--amber)'
    return 'var(--red)'
  }

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="var(--border)"
          strokeWidth="8"
          fill="none"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={getColor()}
          strokeWidth="8"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center" style={{ width: size, height: size }}>
        <span
          className={clsx(
            'font-mono text-2xl font-bold',
            percentage >= 80 ? 'text-[var(--red)]' : 'text-[var(--text-primary)]'
          )}
        >
          {current.toFixed(2)}%
        </span>
        <span className="text-xs text-[var(--text-muted)]">of {max}%</span>
      </div>
    </div>
  )
}
