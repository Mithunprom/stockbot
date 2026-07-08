import { View, Text, StyleSheet } from 'react-native';
import { colors, spacing, fontSize } from '../utils/theme';

export function HeatGauge({ heat, limit = 0.8 }) {
  const pct = Math.min(heat || 0, 1);
  const overLimit = pct >= limit;
  const barColor = overLimit ? colors.red : pct > 0.6 ? colors.orange : colors.green;

  return (
    <View style={styles.container}>
      <View style={styles.labelRow}>
        <Text style={styles.label}>Portfolio Heat</Text>
        <Text style={[styles.value, { color: barColor }]}>{(pct * 100).toFixed(1)}%</Text>
      </View>
      <View style={styles.bar}>
        <View style={[styles.fill, { width: `${pct * 100}%`, backgroundColor: barColor }]} />
        <View style={[styles.limitLine, { left: `${limit * 100}%` }]} />
      </View>
      <View style={styles.legendRow}>
        <Text style={styles.legend}>0%</Text>
        <Text style={[styles.legend, { color: colors.orange }]}>{(limit * 100).toFixed(0)}% limit</Text>
        <Text style={styles.legend}>100%</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: spacing.sm,
  },
  labelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  label: {
    color: colors.textSecondary,
    fontSize: fontSize.sm,
  },
  value: {
    fontSize: fontSize.sm,
    fontWeight: '700',
  },
  bar: {
    height: 8,
    backgroundColor: colors.surface,
    borderRadius: 4,
    position: 'relative',
    overflow: 'visible',
  },
  fill: {
    height: '100%',
    borderRadius: 4,
  },
  limitLine: {
    position: 'absolute',
    top: -2,
    width: 2,
    height: 12,
    backgroundColor: colors.orange,
  },
  legendRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 2,
  },
  legend: {
    color: colors.textMuted,
    fontSize: fontSize.xs,
  },
});
