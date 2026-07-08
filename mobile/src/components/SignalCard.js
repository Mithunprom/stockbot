import { View, Text, StyleSheet } from 'react-native';
import { colors, spacing, fontSize } from '../utils/theme';
import { fmt } from '../utils/format';

const strengthColor = {
  strong: colors.strong,
  moderate: colors.moderate,
  weak: colors.weak,
  flat: colors.flat,
};

export function SignalCard({ signal }) {
  const dir = signal.lgbm_direction > 0 ? 'LONG' : signal.lgbm_direction < 0 ? 'SHORT' : 'FLAT';
  const dirColor = dir === 'LONG' ? colors.green : dir === 'SHORT' ? colors.red : colors.textMuted;
  const sc = strengthColor[signal.strength] || colors.textMuted;

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text style={styles.ticker}>{signal.ticker}</Text>
        <View style={[styles.strengthBadge, { backgroundColor: sc + '22', borderColor: sc }]}>
          <Text style={[styles.strengthText, { color: sc }]}>{signal.strength.toUpperCase()}</Text>
        </View>
      </View>

      <View style={styles.row}>
        <View style={styles.metric}>
          <Text style={styles.label}>Signal</Text>
          <Text style={[styles.value, { color: signal.ensemble_signal >= 0 ? colors.green : colors.red }]}>
            {fmt.sig(signal.ensemble_signal)}
          </Text>
        </View>
        <View style={styles.metric}>
          <Text style={styles.label}>Direction</Text>
          <Text style={[styles.value, { color: dirColor }]}>{dir}</Text>
        </View>
        <View style={styles.metric}>
          <Text style={styles.label}>Confidence</Text>
          <Text style={styles.value}>{(signal.lgbm_confidence * 100).toFixed(1)}%</Text>
        </View>
      </View>

      <View style={styles.row}>
        <View style={styles.metric}>
          <Text style={styles.label}>Pred Return</Text>
          <Text style={[styles.value, { color: signal.lgbm_pred_return >= 0 ? colors.green : colors.red }]}>
            {fmt.pct(signal.lgbm_pred_return)}
          </Text>
        </View>
        <View style={styles.metric}>
          <Text style={styles.label}>Dir Prob</Text>
          <Text style={styles.value}>{(signal.lgbm_dir_prob * 100).toFixed(1)}%</Text>
        </View>
        <View style={styles.metric}>
          <Text style={styles.label}>LightGBM</Text>
          <Text style={[styles.value, { color: colors.blue }]}>{(signal.weights?.lgbm * 100).toFixed(0)}%</Text>
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
  ticker: {
    color: colors.text,
    fontSize: fontSize.lg,
    fontWeight: '700',
  },
  strengthBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: 6,
    borderWidth: 1,
  },
  strengthText: {
    fontSize: fontSize.xs,
    fontWeight: '600',
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: spacing.xs,
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
    fontSize: fontSize.md,
    fontWeight: '600',
  },
});
