import { useEffect, useState, useCallback } from 'react';
import { View, Text, StyleSheet, ScrollView, RefreshControl } from 'react-native';
import { colors, spacing, fontSize } from '../../src/utils/theme';
import { api } from '../../src/utils/api';
import { useStockBotWS } from '../../src/hooks/useStockBotWS';
import { fmt } from '../../src/utils/format';

export default function RiskScreen() {
  const ws = useStockBotWS();
  const [summary, setSummary] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const data = await api.portfolioSummary();
      setSummary(data);
    } catch (_) {}
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
  }, [fetchData]);

  const heat = ws.portfolioHeat ?? summary?.portfolio_heat ?? 0;
  const halted = ws.halted || summary?.halted;
  const haltReason = ws.haltReason || summary?.halt_reason;
  const drawdown = summary?.drawdown_pct ?? 0;
  const dailyPnlPct = summary?.daily_pnl_pct ?? 0;
  const consecutiveLosses = summary?.consecutive_losses ?? 0;
  const openPositions = summary?.n_open_positions ?? 0;
  const tradesToday = summary?.n_trades_today ?? 0;

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.blue} />}
    >
      <Text style={styles.title}>Risk Dashboard</Text>

      {/* Circuit Breaker Status */}
      <View style={[styles.circuitCard, halted ? styles.circuitHalted : styles.circuitOk]}>
        <View style={styles.circuitHeader}>
          <Text style={styles.circuitIcon}>{halted ? '\u26D4' : '\u2705'}</Text>
          <Text style={[styles.circuitTitle, { color: halted ? colors.red : colors.green }]}>
            {halted ? 'TRADING HALTED' : 'TRADING ACTIVE'}
          </Text>
        </View>
        {halted && haltReason && (
          <Text style={styles.circuitReason}>{haltReason}</Text>
        )}
        {!halted && (
          <Text style={styles.circuitSubtext}>All circuit breakers normal</Text>
        )}
      </View>

      {/* Portfolio Heat Gauge */}
      <View style={styles.gaugeCard}>
        <Text style={styles.gaugeTitle}>Portfolio Heat</Text>
        <Text style={styles.gaugeSubtext}>Exposure as % of max allowed</Text>
        <View style={styles.gaugeContainer}>
          <View style={styles.gaugeTrack}>
            <View style={[styles.gaugeFill, {
              width: `${Math.min(heat * 100, 100)}%`,
              backgroundColor: heatColor(heat),
            }]} />
            {/* 80% limit marker */}
            <View style={styles.gaugeLimitLine} />
          </View>
          <View style={styles.gaugeLabels}>
            <Text style={styles.gaugeLabelText}>0%</Text>
            <Text style={[styles.gaugeLabelText, styles.gaugeLimitLabel]}>80%</Text>
            <Text style={styles.gaugeLabelText}>100%</Text>
          </View>
        </View>
        <Text style={[styles.gaugeValue, { color: heatColor(heat) }]}>
          {(heat * 100).toFixed(1)}%
        </Text>
      </View>

      {/* Daily Loss Gauge */}
      <View style={styles.gaugeCard}>
        <Text style={styles.gaugeTitle}>Daily P&L</Text>
        <Text style={styles.gaugeSubtext}>Daily loss limit: -2.0%</Text>
        <View style={styles.gaugeContainer}>
          <View style={styles.gaugeTrack}>
            <View style={[styles.gaugeFill, {
              width: `${Math.min(Math.abs(dailyPnlPct) / 2 * 100, 100)}%`,
              backgroundColor: dailyPnlPct >= 0 ? colors.green : lossColor(dailyPnlPct),
            }]} />
          </View>
        </View>
        <Text style={[styles.gaugeValue, {
          color: dailyPnlPct >= 0 ? colors.green : lossColor(dailyPnlPct),
        }]}>
          {dailyPnlPct >= 0 ? '+' : ''}{dailyPnlPct.toFixed(2)}%
        </Text>
      </View>

      {/* Drawdown Gauge */}
      <View style={styles.gaugeCard}>
        <Text style={styles.gaugeTitle}>Drawdown</Text>
        <Text style={styles.gaugeSubtext}>Max allowed: 8.0%</Text>
        <View style={styles.gaugeContainer}>
          <View style={styles.gaugeTrack}>
            <View style={[styles.gaugeFill, {
              width: `${Math.min(drawdown / 8 * 100, 100)}%`,
              backgroundColor: drawdownColor(drawdown),
            }]} />
            {/* 5% warning marker */}
            <View style={[styles.gaugeLimitLine, { left: '62.5%' }]} />
          </View>
          <View style={styles.gaugeLabels}>
            <Text style={styles.gaugeLabelText}>0%</Text>
            <Text style={[styles.gaugeLabelText, { position: 'absolute', left: '59%' }]}>5%</Text>
            <Text style={styles.gaugeLabelText}>8%</Text>
          </View>
        </View>
        <Text style={[styles.gaugeValue, { color: drawdownColor(drawdown) }]}>
          -{drawdown.toFixed(2)}%
        </Text>
      </View>

      {/* Risk Metrics Grid */}
      <Text style={styles.sectionTitle}>Risk Metrics</Text>
      <View style={styles.metricsGrid}>
        <RiskMetric
          label="Consecutive Losses"
          value={String(consecutiveLosses)}
          color={consecutiveLosses >= 3 ? colors.red : consecutiveLosses >= 2 ? colors.orange : colors.green}
          threshold="Halt at 5"
        />
        <RiskMetric
          label="Open Positions"
          value={String(openPositions)}
          color={openPositions >= 8 ? colors.orange : colors.blue}
          threshold="Max 10"
        />
        <RiskMetric
          label="Trades Today"
          value={String(tradesToday)}
          color={tradesToday >= 15 ? colors.orange : colors.text}
          threshold="Limit 20"
        />
        <RiskMetric
          label="Portfolio Heat"
          value={`${(heat * 100).toFixed(0)}%`}
          color={heatColor(heat)}
          threshold="Limit 80%"
        />
      </View>

      {/* Risk Rules */}
      <Text style={styles.sectionTitle}>Active Risk Rules</Text>
      <View style={styles.rulesCard}>
        <RuleRow label="Daily loss limit" value="-2.0%" active={Math.abs(dailyPnlPct) < 2} />
        <RuleRow label="Max drawdown" value="8.0%" active={drawdown < 8} />
        <RuleRow label="Max heat" value="80%" active={heat < 0.8} />
        <RuleRow label="Stop loss per trade" value="2.0%" active />
        <RuleRow label="Take profit" value="3.5%" active />
        <RuleRow label="Trailing stop" value="2.5%" active />
        <RuleRow label="Max hold time" value="45 bars" active />
        <RuleRow label="Loss streak halt" value="5 consecutive" active={consecutiveLosses < 5} />
      </View>

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

function RiskMetric({ label, value, color, threshold }) {
  return (
    <View style={styles.metricBox}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={[styles.metricValue, { color }]}>{value}</Text>
      <Text style={styles.metricThreshold}>{threshold}</Text>
    </View>
  );
}

function RuleRow({ label, value, active }) {
  return (
    <View style={styles.ruleRow}>
      <View style={[styles.ruleDot, { backgroundColor: active ? colors.green : colors.red }]} />
      <Text style={styles.ruleLabel}>{label}</Text>
      <Text style={styles.ruleValue}>{value}</Text>
    </View>
  );
}

function heatColor(heat) {
  if (heat >= 0.8) return colors.red;
  if (heat >= 0.5) return colors.orange;
  return colors.green;
}

function lossColor(pct) {
  if (pct <= -1.5) return colors.red;
  if (pct <= -0.75) return colors.orange;
  return colors.yellow;
}

function drawdownColor(dd) {
  if (dd >= 5) return colors.red;
  if (dd >= 3) return colors.orange;
  return colors.green;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
    padding: spacing.md,
    paddingTop: 60,
  },
  title: {
    color: colors.text,
    fontSize: fontSize.xl,
    fontWeight: '700',
    marginBottom: spacing.md,
  },
  circuitCard: {
    borderRadius: 12,
    borderWidth: 1,
    padding: spacing.md,
    marginBottom: spacing.md,
    alignItems: 'center',
  },
  circuitHalted: {
    backgroundColor: colors.redDim,
    borderColor: colors.red,
  },
  circuitOk: {
    backgroundColor: colors.greenDim,
    borderColor: colors.green + '66',
  },
  circuitHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  circuitIcon: {
    fontSize: 20,
  },
  circuitTitle: {
    fontSize: fontSize.lg,
    fontWeight: '700',
  },
  circuitReason: {
    color: colors.text,
    fontSize: fontSize.sm,
    marginTop: spacing.xs,
  },
  circuitSubtext: {
    color: colors.textSecondary,
    fontSize: fontSize.sm,
    marginTop: spacing.xs,
  },
  gaugeCard: {
    backgroundColor: colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.cardBorder,
    padding: spacing.md,
    marginBottom: spacing.sm,
  },
  gaugeTitle: {
    color: colors.text,
    fontSize: fontSize.md,
    fontWeight: '700',
  },
  gaugeSubtext: {
    color: colors.textMuted,
    fontSize: fontSize.xs,
    marginBottom: spacing.sm,
  },
  gaugeContainer: {
    marginBottom: spacing.xs,
  },
  gaugeTrack: {
    height: 12,
    backgroundColor: colors.surface,
    borderRadius: 6,
    overflow: 'hidden',
    position: 'relative',
  },
  gaugeFill: {
    height: '100%',
    borderRadius: 6,
  },
  gaugeLimitLine: {
    position: 'absolute',
    left: '80%',
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: colors.yellow,
  },
  gaugeLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 2,
  },
  gaugeLabelText: {
    color: colors.textMuted,
    fontSize: 9,
  },
  gaugeLimitLabel: {
    position: 'absolute',
    left: '76%',
  },
  gaugeValue: {
    fontSize: fontSize.xl,
    fontWeight: '700',
    textAlign: 'center',
  },
  sectionTitle: {
    color: colors.text,
    fontSize: fontSize.md,
    fontWeight: '700',
    marginTop: spacing.sm,
    marginBottom: spacing.sm,
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
    alignItems: 'center',
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
  metricThreshold: {
    color: colors.textMuted,
    fontSize: 9,
    marginTop: 2,
  },
  rulesCard: {
    backgroundColor: colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.cardBorder,
    padding: spacing.md,
  },
  ruleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: colors.cardBorder,
  },
  ruleDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: spacing.sm,
  },
  ruleLabel: {
    color: colors.textSecondary,
    fontSize: fontSize.sm,
    flex: 1,
  },
  ruleValue: {
    color: colors.text,
    fontSize: fontSize.sm,
    fontWeight: '600',
  },
});
