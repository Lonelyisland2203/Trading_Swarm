import { useState, useEffect, useCallback, useRef } from 'react'
import type { LiveUpdate } from '@/types'

const WS_URL = import.meta.env.VITE_WS_URL || `ws://${window.location.host}/ws/live`

interface UseWebSocketResult {
  data: LiveUpdate | null
  isConnected: boolean
  error: string | null
  reconnect: () => void
}

export function useWebSocket(): UseWebSocketResult {
  const [data, setData] = useState<LiveUpdate | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttempts = useRef(0)
  const reconnectTimeoutRef = useRef<number | null>(null)

  const maxReconnectAttempts = 10
  const baseReconnectDelay = 1000

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    try {
      wsRef.current = new WebSocket(WS_URL)

      wsRef.current.onopen = () => {
        setIsConnected(true)
        setError(null)
        reconnectAttempts.current = 0
      }

      wsRef.current.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data) as LiveUpdate
          setData(parsed)
        } catch {
          console.error('Failed to parse WebSocket message')
        }
      }

      wsRef.current.onerror = () => {
        setError('WebSocket connection error')
      }

      wsRef.current.onclose = () => {
        setIsConnected(false)

        // Auto-reconnect with exponential backoff
        if (reconnectAttempts.current < maxReconnectAttempts) {
          const delay = baseReconnectDelay * Math.pow(2, reconnectAttempts.current)
          reconnectAttempts.current++

          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect()
          }, delay)
        } else {
          setError('Max reconnection attempts reached')
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection failed')
    }
  }, [])

  const reconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    reconnectAttempts.current = 0
    wsRef.current?.close()
    connect()
  }, [connect])

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      wsRef.current?.close()
    }
  }, [connect])

  return { data, isConnected, error, reconnect }
}
