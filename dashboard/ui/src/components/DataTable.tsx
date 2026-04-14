import { clsx } from 'clsx'
import type { ReactNode } from 'react'

export interface Column<T> {
  key: keyof T | string
  header: string
  render?: (row: T) => ReactNode
  align?: 'left' | 'center' | 'right'
  mono?: boolean
}

interface DataTableProps<T> {
  columns: Column<T>[]
  data: T[]
  keyField: keyof T
  onRowClick?: (row: T) => void
  selectedKey?: string | number
  rowClassName?: (row: T) => string
  emptyMessage?: string
  maxHeight?: string
}

export function DataTable<T extends object>({
  columns,
  data,
  keyField,
  onRowClick,
  selectedKey,
  rowClassName,
  emptyMessage = 'No data',
  maxHeight = '400px',
}: DataTableProps<T>) {
  return (
    <div className="overflow-auto rounded border border-[var(--border)]" style={{ maxHeight }}>
      <table className="w-full border-collapse">
        <thead className="sticky top-0 bg-[var(--bg-surface)]">
          <tr>
            {columns.map((col) => (
              <th
                key={String(col.key)}
                className={clsx(
                  'px-3 py-2 text-xs uppercase tracking-wider text-[var(--text-muted)] border-b border-[var(--border)]',
                  col.align === 'right' && 'text-right',
                  col.align === 'center' && 'text-center',
                  (!col.align || col.align === 'left') && 'text-left'
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="px-3 py-8 text-center text-[var(--text-muted)]"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            data.map((row) => {
              const key = row[keyField]
              const isSelected = selectedKey !== undefined && key === selectedKey
              return (
                <tr
                  key={String(key)}
                  onClick={() => onRowClick?.(row)}
                  className={clsx(
                    'border-b border-[var(--border-subtle)]',
                    onRowClick && 'cursor-pointer hover:bg-[var(--bg-elevated)]',
                    isSelected && 'bg-[var(--bg-elevated)]',
                    rowClassName?.(row)
                  )}
                >
                  {columns.map((col) => {
                    const value = col.render ? col.render(row) : String(row[col.key as keyof T] ?? '')
                    return (
                      <td
                        key={String(col.key)}
                        className={clsx(
                          'px-3 py-2 text-sm',
                          col.align === 'right' && 'text-right',
                          col.align === 'center' && 'text-center',
                          col.mono && 'font-mono'
                        )}
                      >
                        {value}
                      </td>
                    )
                  })}
                </tr>
              )
            })
          )}
        </tbody>
      </table>
    </div>
  )
}
