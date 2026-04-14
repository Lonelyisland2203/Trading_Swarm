import { useState } from 'react'
import { TabNav } from '@/components'
import {
  LiveOverview,
  SignalPipeline,
  Performance,
  XGBoostModel,
  Autoresearch,
  RiskMonitor,
} from '@/pages'
import type { Tab, TabId } from '@/types'

const TABS: Tab[] = [
  { id: 'live', label: 'Live Overview' },
  { id: 'signals', label: 'Signal Pipeline' },
  { id: 'performance', label: 'Performance' },
  { id: 'xgboost', label: 'XGBoost Model' },
  { id: 'autoresearch', label: 'Autoresearch' },
  { id: 'risk', label: 'Risk Monitor' },
]

function App() {
  const [activeTab, setActiveTab] = useState<TabId>('live')

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-[var(--accent)] flex items-center justify-center">
            <span className="text-white font-bold text-sm">TS</span>
          </div>
          <h1 className="text-lg font-semibold text-[var(--text-primary)]">
            Trade Swarm
          </h1>
        </div>
        <div className="flex items-center gap-4 text-xs text-[var(--text-muted)]">
          <span>READ-ONLY</span>
          <span className="font-mono">{new Date().toLocaleDateString()}</span>
        </div>
      </header>

      {/* Tab Navigation */}
      <TabNav tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {activeTab === 'live' && <LiveOverview />}
        {activeTab === 'signals' && <SignalPipeline />}
        {activeTab === 'performance' && <Performance />}
        {activeTab === 'xgboost' && <XGBoostModel />}
        {activeTab === 'autoresearch' && <Autoresearch />}
        {activeTab === 'risk' && <RiskMonitor />}
      </main>

      {/* Footer */}
      <footer className="px-6 py-2 border-t border-[var(--border)] text-xs text-[var(--text-muted)] flex items-center justify-between">
        <span>Dashboard is read-only. No execution controls.</span>
        <span className="font-mono">localhost:8420</span>
      </footer>
    </div>
  )
}

export default App
