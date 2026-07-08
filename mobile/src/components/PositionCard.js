import { View, Text, StyleSheet } from 'react-native';
import { colors, spacing, fontSize } from '../utils/theme';
import { fmt } from '../utils/format';

export function PositionCard({ position }) {
  const pnlColor = position.unrealized_pnl >= 0 ? colors.green : colors.red;
  const isLong = position.side === 'long';

  // Progress bar: entry -> stop_loss / take_profit with current price marker
  const entry = position.avg_entry_price;
  const sl = position.stop_loss_price;
  const tp = position.take_profit_price;
  const current = position.last_price;
  const range = tp - sl;
  const progress = range !== 0 ? Math.max(0, Math.min(1, (current - sl) / range)) : 0.5;

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View style={styles.tickerRow}>
          <Text style={styles.ticker}>{position.ticker}</Text>
          <View style={[styles.sideBadge, { backgroundColor: isLong ? colors.greenDim : colors.redDim }]}>
            <Text style={[styles.sideText, { color: isLong ? colors.green : colors.red }]}>
              {position.side.toUpperCase()}
            </Text>
          </View>
        </View>
        <Text style={[styles.pnl, { color: pnlColor }]}>
          {fmt.dollar(position.unrealized_pnl)}
        </Text>
      </View>

      <View style={styles.row}>
        <View style={styles.metric}>
          <Text style={styles.label}>Qty</Text>
          <Text style={styles.value}>{fmt.qty(position.qty)}</Text>
        </View>
        <View style={styles.metric}>
          <Text style={styles.label}>Entry</Text>
          <Text style={styles.value}>{fmt.price(entry)}</Text>
        </View>
        <View style={styles.metric}>
          <Text style={styles.label}>Current</Text>
          <Text style={[styles.value, { color: pnlColor }]}>{fmt.price(current)}</Text>
        </View>
        <View style={styles.metric}>
          <Text style={styles.label}>PnL %</Text>
          <Text style={[styles.value, { color: pnlColor }]}>{fmt.pct(position.unrealized_pnl_pct)}</Text>
        </View>
      </View>

      {/* Price level progress bar */}
      <View style={styles.progressContainer}>
        <View style={styles.progressLabels}>
          <Text style={[styles.levelLabel, { color: colors.red }]}>SL {fmt.price(sl)}</Text>
          <Text style={[styles.levelLabel, { color: colors.textMuted }]}>Entry {fmt.price(entry)}</Text>
          <Text style={[styles.levelLabel, { color: colors.green }]}>TP {fmt.price(tp)}</Text>
        </View>
        <View style={styles.progressBar}>
          <View style={[styles.progressFill, {
            width: `${progress * 100}%`,
            backgroundColor: progress > 0.5 ? colors.green : colors.red,
          }]} />
          <View style={[styles.progressMarker, { left: `${progress * 100}%` }]} />
        </View>
      </View>

      {/* Exit info */}
      <View style={styles.exitRow}>
        <View style={styles.exitMetric}>
          <Text style={styles.label}>Trailing Stop</Text>
          <Text style={[styles.exitValue, { color: colors.orange }]}>{fmt.price(position.trailing_stop_price)}</Text>
        </View>
        <View style={styles.exitMetric}>
          <Text style={styles.label}>Peak</Text>
          <Text style={styles.exitValue}>{fmt.price(position.peak_price)}</Text>
        </View>
        <View style={styles.exitMetric}>
          <Text style={styles.label}>Hold</Text>
          <Text style={[styles.exitValue, {
            color: position.bars_remaining <= 5 ? colors.red : colors.text,
          }]}>
            {position.bars_remaining} bars left
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.cardBorder,
    padding: spacing.md,
    marginBottom: spacing.sm,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  tickerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  ticker: {
    color: colors.text,
    fontSize: fontSize.lg,
    fontWeight: '700',
  },
  sideBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: 4,
  },
  sideText: {
    fontSize: fontSize.xs,
    fontWeight: '700',
  },
  pnl: {
    fontSize: fontSize.lg,
    fontWeight: '700',
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.sm,
  },
  metric: {
    flex: 1,
    alignItems: 'center',
  },
  label: {
    color: colors.textMuted,
    fontSize: fontSize.xs,
    marginBottom: 2,
  },
  value: {
    color: colors.text,
    fontSize: fontSize.sm,
    fontWeight: '600',
  },
  progressContainer: {
    marginVertical: spacing.sm,
  },
  progressLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  levelLabel: {
    fontSize: fontSize.xs,
    fontWeight: '500',
  },
  progressBar: {
    height: 6,
    backgroundColor: colors.surface,
    borderRadius: 3,
    position: 'relative',
    overflow: 'visible',
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
    opacity: 0.6,
  },
  progressMarker: {
    position: 'absolute',
    top: -3,
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: colors.white,
    marginLeft: -6,
    borderWidth: 2,
    borderColor: colors.blue,
  },
  exitRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: spacing.xs,
  },
  exitMetric: {
    flex: 1,
    alignItems: 'center',
  },
  exitValue: {
    color: colors.text,
    fontSize: fontSize.sm,
    fontWeight: '500',
  },
});
