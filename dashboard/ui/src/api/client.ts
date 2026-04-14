import type {
  Position,
  Signal,
  DailyPnL,
  PerformanceData,
  FeatureImportance,
  XGBoostMetric,
  AutoresearchExperiment,
  FundingRates,
  OpenInterest,
  HealthStatus,
} from '@/types'

const BASE_URL = import.meta.env.VITE_API_URL || ''

async function fetchJSON<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${endpoint}`)
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }
  return response.json()
}

export async function getPositions(): Promise<Position[]> {
  return fetchJSON<Position[]>('/api/positions')
}

export async function getDailyPnL(): Promise<DailyPnL[]> {
  const data = await fetchJSON<{ daily_pnl: DailyPnL[] }>('/api/pnl/daily')
  return data.daily_pnl
}

export async function getRecentSignals(): Promise<Signal[]> {
  return fetchJSON<Signal[]>('/api/signals/recent')
}

export async function getPerformance(): Promise<PerformanceData> {
  return fetchJSON<PerformanceData>('/api/performance')
}

export async function getXGBoostFeatures(): Promise<FeatureImportance[]> {
  const data = await fetchJSON<{ features: FeatureImportance[] }>('/api/xgboost/features')
  return data.features
}

export async function getXGBoostMetrics(): Promise<XGBoostMetric[]> {
  const data = await fetchJSON<{ metrics: XGBoostMetric[] }>('/api/xgboost/metrics')
  return data.metrics
}

export async function getAutoresearchLog(): Promise<AutoresearchExperiment[]> {
  return fetchJSON<AutoresearchExperiment[]>('/api/autoresearch/log')
}

export async function getFundingRates(): Promise<FundingRates> {
  return fetchJSON<FundingRates>('/api/risk/funding')
}

export async function getOpenInterest(): Promise<OpenInterest> {
  return fetchJSON<OpenInterest>('/api/risk/oi')
}

export async function getHealth(): Promise<HealthStatus> {
  return fetchJSON<HealthStatus>('/api/health')
}
