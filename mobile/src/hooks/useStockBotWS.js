// WebSocket hook for real-time StockBot data
import { useEffect, useRef, useState, useCallback } from 'react';
import { AppState } from 'react-native';
import { WS_URL } from '../utils/config';

export function useStockBotWS() {
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const [connected, setConnected] = useState(false);
  const [signals, setSignals] = useState([]);
  const [positions, setPositions] = useState({});
  const [positionsDetail, setPositionsDetail] = useState([]);
  const [portfolioValue, setPortfolioValue] = useState(null);
  const [portfolioHeat, setPortfolioHeat] = useState(null);
  const [halted, setHalted] = useState(false);
  const [haltReason, setHaltReason] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'signals') {
            if (msg.signals?.length) setSignals(msg.signals);
            if (msg.positions) setPositions(msg.positions);
            if (msg.positions_detail) setPositionsDetail(msg.positions_detail);
            if (msg.portfolio_value != null) setPortfolioValue(msg.portfolio_value);
            if (msg.portfolio_heat != null) setPortfolioHeat(msg.portfolio_heat);
            if (msg.halted != null) setHalted(msg.halted);
            if (msg.halt_reason) setHaltReason(msg.halt_reason);
            setLastUpdate(new Date());
          }
        } catch (_) { /* ignore malformed */ }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimer.current = setTimeout(connect, 5000);
      };

      ws.onerror = () => ws.close();
    } catch (_) {
      reconnectTimer.current = setTimeout(connect, 5000);
    }
  }, []);

  const disconnect = useCallback(() => {
    clearTimeout(reconnectTimer.current);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  // Disconnect when app goes to background, reconnect when foregrounded
  useEffect(() => {
    const sub = AppState.addEventListener('change', (state) => {
      if (state === 'active') connect();
      else if (state === 'background') disconnect();
    });
    return () => sub.remove();
  }, [connect, disconnect]);

  return {
    connected,
    signals,
    positions,
    positionsDetail,
    portfolioValue,
    portfolioHeat,
    halted,
    haltReason,
    lastUpdate,
  };
}
