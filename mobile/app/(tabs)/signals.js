import { useEffect, useState, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, RefreshControl, TouchableOpacity } from 'react-native';
import { colors, spacing, fontSize } from '../../src/utils/theme';
import { api } from '../../src/utils/api';
import { useStockBotWS } from '../../src/hooks/useStockBotWS';
import { SignalCard } from '../../src/components/SignalCard';
import { fmt } from '../../src/utils/format';

export default function SignalsScreen() {
  const ws = useStockBotWS();
  const [actionable, setActionable] = useState([]);
  const [refreshing, setRefreshing] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const fetchActionable = useCallback(async () => {
    try {
      const data = await api.signalsActionable();
      setActionable(data.signals || []);
    } catch (_) {}
  }, []);

  useEffect(() => {
    fetchActionable();
    const interval = setInterval(fetchActionable, 30000);
    return () => clearInterval(interval);
  }, [fetchActionable]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchActionable();
    setRefreshing(false);
  }, [fetchActionable]);

  // Combine: actionable signals first, then all WS signals
  const allSignals = ws.signals || [];
  const displaySignals = showAll ? allSignals : allSignals.filter(s => Math.abs(s.ensemble_signal) >= 0.3);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Live Signals</Text>
        <TouchableOpacity onPress={() => setShowAll(!showAll)}>
          <Text style={styles.filterToggle}>{showAll ? 'Strong Only' : 'Show All'}</Text>
        </TouchableOpacity>
      </View>

      {/* Actionable signals banner */}
      {actionable.length > 0 && (
        <View style={styles.actionableBanner}>
          <Text style={styles.actionableTitle}>Actionable ({actionable.length})</Text>
          {actionable.map((sig) => (
            <View key={sig.ticker} style={styles.actionableItem}>
              <Text style={styles.actionableTicker}>{sig.ticker}</Text>
              <Text style={[styles.actionableSide, {
                color: sig.recommended_side === 'buy' ? colors.green : colors.red,
              }]}>
                {sig.recommended_side.toUpperCase()}
              </Text>
              <Text style={styles.actionableSize}>{fmt.price(sig.recommended_notional)}</Text>
              <Text style={[styles.actionableStrength, {
                color: sig.strength === 'strong' ? colors.green : sig.strength === 'moderate' ? colors.blue : colors.yellow,
              }]}>
                {sig.strength}
              </Text>
            </View>
          ))}
        </View>
      )}

      <FlatList
        data={displaySignals}
        keyExtractor={(item) => item.ticker}
        renderItem={({ item }) => <SignalCard signal={item} />}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.blue} />
        }
        ListEmptyComponent={
          <Text style={styles.emptyText}>Waiting for signals...</Text>
        }
        contentContainerStyle={{ paddingBottom: 100 }}
      />
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  title: {
    color: colors.text,
    fontSize: fontSize.xl,
    fontWeight: '700',
  },
  filterToggle: {
    color: colors.blue,
    fontSize: fontSize.sm,
    fontWeight: '600',
  },
  actionableBanner: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.green + '44',
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  actionableTitle: {
    color: colors.green,
    fontSize: fontSize.md,
    fontWeight: '700',
    marginBottom: spacing.sm,
  },
  actionableItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
  },
  actionableTicker: {
    color: colors.text,
    fontSize: fontSize.md,
    fontWeight: '600',
    width: 80,
  },
  actionableSide: {
    fontSize: fontSize.sm,
    fontWeight: '700',
    width: 40,
  },
  actionableSize: {
    color: colors.textSecondary,
    fontSize: fontSize.sm,
    width: 80,
    textAlign: 'right',
  },
  actionableStrength: {
    fontSize: fontSize.xs,
    fontWeight: '600',
    width: 60,
    textAlign: 'right',
  },
  emptyText: {
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: 40,
    fontSize: fontSize.md,
  },
});
