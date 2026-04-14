import { clsx } from 'clsx'
import type { Tab, TabId } from '@/types'

interface TabNavProps {
  tabs: Tab[]
  activeTab: TabId
  onTabChange: (tab: TabId) => void
}

export function TabNav({ tabs, activeTab, onTabChange }: TabNavProps) {
  return (
    <nav className="flex gap-1 border-b border-[var(--border)] px-4">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={clsx(
            'px-4 py-3 text-sm font-medium transition-colors relative',
            activeTab === tab.id
              ? 'text-[var(--text-primary)]'
              : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
          )}
        >
          {tab.label}
          {activeTab === tab.id && (
            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--accent)]" />
          )}
        </button>
      ))}
    </nav>
  )
}
