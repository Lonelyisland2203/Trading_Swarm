// API Response Types

export interface Position {
  symbol: string
  side: 'LONG' | 'SHORT'
  amount: number
  entry_price: number
  mark_price: number
  unrealized_pnl: number
  leverage: number
}

export interface Signal {
  timestamp: string
  symbol: string
  xgboost_prob?: number
  xgboost_direction?: 'LONG' | 'SHORT' | 'FLAT'
  llm_context?: LLMContext
  synthesis_decision?: 'LONG' | 'SHORT' | 'FLAT'
  veto?: boolean
  outcome_pnl?: number | null
  outcome_resolved?: boolean
}

export interface LLMContext {
  bullish_factors: string[]
  bearish_factors: string[]
  regime_flag: string
  confidence: number
}

export interface DailyPnL {
  date: string
  pnl: number
  cumulative: number
}

export interface EquityCurvePoint {
  timestamp: string
  equity: number
}

export interface RollingSharpePoint {
  timestamp: string
  sharpe: number
}

export interface DrawdownPoint {
  timestamp: string
  drawdown: number
}

export interface PerformanceData {
  equity_curve: EquityCurvePoint[]
  rolling_sharpe: RollingSharpePoint[]
  drawdown: DrawdownPoint[]
  win_rate: number
}

export interface FeatureImportance {
  feature: string
  importance: number
}

export interface XGBoostMetric {
  timestamp: string
  ic: number
  brier: number
  sharpe_net: number
}

export interface AutoresearchExperiment {
  experiment_id: string
  timestamp: string
  change_description: string
  sharpe_net: number
  ic: number
  brier: number
  accuracy: number
  false_bullish_rate: number
  kept_or_reverted: 'kept' | 'reverted'
}

export interface FundingRates {
  [symbol: string]: number
}

export interface OpenInterest {
  [symbol: string]: {
    oi: number
    timestamp: number
  }
}

export interface HealthStatus {
  watchdog_heartbeat?: string
  kill_switch_active?: boolean
  signal_loop_status?: 'running' | 'stopped' | 'error'
  last_signal?: string
  daily_loss_pct?: number
}

export interface LiveUpdate {
  positions: Position[]
  latest_signal: Signal | null
  daily_pnl: DailyPnL[]
  health: HealthStatus
  timestamp: string
}

// UI Types
export type TabId = 'live' | 'signals' | 'performance' | 'xgboost' | 'autoresearch' | 'risk'

export interface Tab {
  id: TabId
  label: string
}
