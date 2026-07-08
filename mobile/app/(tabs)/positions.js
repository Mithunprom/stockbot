import { useEffect, useState, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, RefreshControl } from 'react-native';
import { colors, spacing, fontSize } from '../../src/utils/theme';
import { api } from '../../src/utils/api';
import { useStockBotWS } from '../../src/hooks/useStockBotWS';
import { PositionCard } from '../../src/components/PositionCard';
import { fmt } from '../../src/utils/format';

export default function PositionsScreen() {
  const ws = useStockBotWS();
  const [detailPositions, setDetailPositions] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  const fetchPositions = useCallback(async () => {
    try {
      const data = await api.positionsDetail();
      setDetailPositions(data.positions || []);
    } catch (_) {}
  }, []);

  useEffect(() => {
    fetchPositions();
    const interval = setInterval(fetchPositions, 15000);
    return () => clearInterval(interval);
  }, [fetchPositions]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchPositions();
    setRefreshing(false);
  }, [fetchPositions]);

  // Use WebSocket detail if available, fall back to REST
  const positions = ws.positionsDetail.length > 0 ? ws.positionsDetail : detailPositions;

  // Total unrealized PnL
  const totalPnl = positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0);
  const pnlColor = totalPnl >= 0 ? colors.green : colors.red;

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Open Positions</Text>
        <Text style={[styles.totalPnl, { color: pnlColor }]}>
          {fmt.dollar(totalPnl)}
        </Text>
      </View>

      <Text style={styles.count}>{positions.length} position{positions.length !== 1 ? 's' : ''}</Text>

      <FlatList
        data={positions}
        keyExtractor={(item) => item.ticker}
        renderItem={({ item }) => <PositionCard position={item} />}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.blue} />
        }
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyIcon}>-</Text>
            <Text style={styles.emptyText}>No open positions</Text>
            <Text style={styles.emptySubtext}>Signals will generate entries during market hours</Text>
          </View>
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
    marginBottom: spacing.xs,
  },
  title: {
    color: colors.text,
    fontSize: fontSize.xl,
    fontWeight: '700',
  },
  totalPnl: {
    fontSize: fontSize.lg,
    fontWeight: '700',
  },
  count: {
    color: colors.textMuted,
    fontSize: fontSize.sm,
    marginBottom: spacing.md,
  },
  emptyContainer: {
    alignItems: 'center',
    marginTop: 60,
  },
  emptyIcon: {
    color: colors.textMuted,
    fontSize: 48,
    marginBottom: spacing.md,
  },
  emptyText: {
    color: colors.textSecondary,
    fontSize: fontSize.lg,
    fontWeight: '600',
  },
  emptySubtext: {
    color: colors.textMuted,
    fontSize: fontSize.sm,
    marginTop: spacing.xs,
  },
});
