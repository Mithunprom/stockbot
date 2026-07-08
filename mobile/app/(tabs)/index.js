import { useEffect, useState, useCallback } from 'react';
import { View, Text, StyleSheet, ScrollView, RefreshControl } from 'react-native';
import { colors, spacing, fontSize } from '../../src/utils/theme';
import { fmt } from '../../src/utils/format';
import { api } from '../../src/utils/api';
import { useStockBotWS } from '../../src/hooks/useStockBotWS';
import { HeatGauge } from '../../src/components/HeatGauge';

export default function DashboardScreen() {
  const ws = useStockBotWS();
  const [summary, setSummary] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchSummary = useCallback(async () => {
    try {
      const data = await api.portfolioSummary();
      setSummary(data);
    } catch (_) {}
  }, []);

  useEffect(() => {
    fetchSummary();
    const interval = setInterval(fetchSummary, 30000);
    return () => clearInterval(interval);
  }, [fetchSummary]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchSummary();
    setRefreshing(false);
  }, [fetchSummary]);

  const pv = ws.portfolioValue || summary?.portfolio_value;
  const heat = ws.portfolioHeat || summary?.portfolio_heat;
  const dailyPnl = summary?.daily_pnl_pct;
  const dailyDollar = summary?.daily_pnl_dollar;
  const pnlColor = (dailyPnl || 0) >= 0 ? colors.green : colors.red;

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.blue} />}
    >
      {/* Status bar */}
      <View style={styles.statusRow}>
        <View style={styles.statusItem}>
          <View style={[styles.dot, { backgroundColor: ws.connected ? colors.connected : colors.disconnected }]} />
          <Text style={styles.statusText}>{ws.connected ? 'LIVE' : 'OFFLINE'}</Text>
        </View>
        <View style={[styles.modeBadge, {
          backgroundColor: summary?.mode === 'live' ? colors.redDim : '#3d3200',
        }]}>
          <Text style={[styles.modeText, {
            color: summary?.mode === 'live' ? colors.red : colors.yellow,
          }]}>
            {(summary?.mode || 'PAPER').toUpperCase()}
          </Text>
        </View>
        <Text style={styles.statusText}>
          {summary?.market_open ? 'Market Open' : 'Market Closed'}
        </Text>
      </View>

      {/* Portfolio value hero */}
      <View style={styles.heroCard}>
        <Text style={styles.heroLabel}>Portfolio Value</Text>
        <Text style={styles.heroValue}>
          {pv != null ? `$${pv.toLocaleString('en-US', { minimumFractionDigits: 2 })}` : '--'}
        </Text>
        <View style={styles.pnlRow}>
          <Text style={[styles.pnlValue, { color: pnlColor }]}>
            {dailyDollar != null ? fmt.dollar(dailyDollar) : '--'}
          </Text>
          <Text style={[styles.pnlPct, { color: pnlColor }]}>
            {dailyPnl != null ? fmt.pctRaw(dailyPnl) : '--'}
          </Text>
          <Text style={styles.pnlLabel}>today</Text>
        </View>
      </View>

      {/* Heat gauge */}
      <View style={styles.section}>
        <HeatGauge heat={heat} />
      </View>

      {/* Key metrics grid */}
      <View style={styles.metricsGrid}>
        <MetricBox label="Unrealized P&L" value={fmt.dollar(summary?.total_unrealized_pnl)}
          color={(summary?.total_unrealized_pnl || 0) >= 0 ? colors.green : colors.red} />
        <MetricBox label="Available Cash" value={fmt.price(summary?.available_cash)} color={colors.text} />
        <MetricBox label="Drawdown" value={summary?.drawdown_pct != null ? `-${summary.drawdown_pct.toFixed(2)}%` : '--'}
          color={summary?.drawdown_pct > 5 ? colors.red : colors.text} />
        <MetricBox label="Open Positions" value={String(summary?.n_open_positions ?? '--')} color={colors.blue} />
        <MetricBox label="Trades Today" value={String(summary?.n_trades_today ?? '--')} color={colors.text} />
        <MetricBox label="Loss Streak" value={String(summary?.consecutive_losses ?? '--')}
          color={(summary?.consecutive_losses || 0) >= 3 ? colors.red : colors.text} />
      </View>

      {/* Halt status */}
      {(ws.halted || summary?.halted) && (
        <View style={styles.haltBanner}>
          <Text style={styles.haltText}>TRADING HALTED</Text>
          <Text style={styles.haltReason}>{ws.haltReason || summary?.halt_reason}</Text>
        </View>
      )}

      {/* Last update */}
      <Text style={styles.lastUpdate}>
        Last update: {ws.lastUpdate ? ws.lastUpdate.toLocaleTimeString() : 'waiting...'}
      </Text>
    </ScrollView>
  );
}

function MetricBox({ label, value, color }) {
  return (
    <View style={styles.metricBox}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={[styles.metricValue, { color }]}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
    padding: spacing.md,
    paddingTop: 60,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  statusItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  statusText: {
    color: colors.textSecondary,
    fontSize: fontSize.sm,
  },
  modeBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: 4,
  },
  modeText: {
    fontSize: fontSize.xs,
    fontWeight: '700',
  },
  heroCard: {
    backgroundColor: colors.card,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.cardBorder,
    padding: spacing.lg,
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  heroLabel: {
    color: colors.textSecondary,
    fontSize: fontSize.sm,
    marginBottom: spacing.xs,
  },
  heroValue: {
    color: colors.text,
    fontSize: fontSize.hero,
    fontWeight: '700',
  },
  pnlRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.sm,
  },
  pnlValue: {
    fontSize: fontSize.lg,
    fontWeight: '600',
  },
  pnlPct: {
    fontSize: fontSize.md,
    fontWeight: '500',
  },
  pnlLabel: {
    color: colors.textMuted,
    fontSize: fontSize.sm,
  },
  section: {
    marginBottom: spacing.md,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  metricBox: {
    backgroundColor: colors.card,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.cardBorder,
    padding: spacing.md,
    width: '48%',
    flexGrow: 1,
  },
  metricLabel: {
    color: colors.textMuted,
    fontSize: fontSize.xs,
    marginBottom: 4,
  },
  metricValue: {
    fontSize: fontSize.xl,
    fontWeight: '700',
  },
  haltBanner: {
    backgroundColor: colors.redDim,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.red,
    padding: spacing.md,
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  haltText: {
    color: colors.red,
    fontSize: fontSize.lg,
    fontWeight: '700',
  },
  haltReason: {
    color: colors.text,
    fontSize: fontSize.sm,
    marginTop: 4,
  },
  lastUpdate: {
    color: colors.textMuted,
    fontSize: fontSize.xs,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
});
