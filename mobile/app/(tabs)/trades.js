import { useEffect, useState, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, RefreshControl } from 'react-native';
import { colors, spacing, fontSize } from '../../src/utils/theme';
import { api } from '../../src/utils/api';
import { fmt } from '../../src/utils/format';

export default function TradesScreen() {
  const [trades, setTrades] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  const fetchTrades = useCallback(async () => {
    try {
      const data = await api.trades(100);
      setTrades(data.trades || []);
    } catch (_) {}
  }, []);

  useEffect(() => {
    fetchTrades();
    const interval = setInterval(fetchTrades, 60000);
    return () => clearInterval(interval);
  }, [fetchTrades]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchTrades();
    setRefreshing(false);
  }, [fetchTrades]);

  // Stats
  const completed = trades.filter(t => t.exit_price);
  const wins = completed.filter(t => t.pnl > 0).length;
  const winRate = completed.length > 0 ? (wins / completed.length * 100).toFixed(1) : '--';
  const totalPnl = completed.reduce((sum, t) => sum + (t.pnl || 0), 0);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Trade History</Text>

      {/* Summary stats */}
      <View style={styles.statsRow}>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>Total Trades</Text>
          <Text style={styles.statValue}>{completed.length}</Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>Win Rate</Text>
          <Text style={[styles.statValue, {
            color: parseFloat(winRate) >= 50 ? colors.green : colors.red,
          }]}>{winRate}%</Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>Total PnL</Text>
          <Text style={[styles.statValue, {
            color: totalPnl >= 0 ? colors.green : colors.red,
          }]}>{fmt.dollar(totalPnl)}</Text>
        </View>
      </View>

      <FlatList
        data={trades}
        keyExtractor={(item, i) => `${item.ticker}-${item.entry_time}-${i}`}
        renderItem={({ item }) => <TradeRow trade={item} />}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.blue} />
        }
        ListEmptyComponent={
          <Text style={styles.emptyText}>No trades yet</Text>
        }
        contentContainerStyle={{ paddingBottom: 100 }}
      />
    </View>
  );
}

function TradeRow({ trade }) {
  const isOpen = !trade.exit_price;
  const pnlColor = (trade.pnl || 0) >= 0 ? colors.green : colors.red;
  const sideColor = trade.side === 'buy' ? colors.green : colors.red;

  const exitReasonLabel = {
    stop_loss: 'Stop Loss',
    take_profit: 'Take Profit',
    trailing_stop: 'Trail Stop',
    max_hold: 'Max Hold',
    signal_reversal: 'Reversal',
  };

  return (
    <View style={styles.tradeCard}>
      <View style={styles.tradeHeader}>
        <View style={styles.tradeTickerRow}>
          <Text style={styles.tradeTicker}>{trade.ticker}</Text>
          <Text style={[styles.tradeSide, { color: sideColor }]}>
            {trade.side?.toUpperCase()}
          </Text>
        </View>
        {!isOpen && (
          <Text style={[styles.tradePnl, { color: pnlColor }]}>
            {fmt.dollar(trade.pnl)} ({fmt.pct(trade.pnl_pct)})
          </Text>
        )}
        {isOpen && <Text style={styles.tradeOpen}>OPEN</Text>}
      </View>

      <View style={styles.tradeDetails}>
        <Text style={styles.tradeDetail}>
          Entry: {fmt.price(trade.entry_price)} @ {trade.entry_time ? new Date(trade.entry_time).toLocaleTimeString() : '--'}
        </Text>
        {trade.exit_price ? (
          <Text style={styles.tradeDetail}>
            Exit: {fmt.price(trade.exit_price)} @ {new Date(trade.exit_time).toLocaleTimeString()}
          </Text>
        ) : null}
      </View>

      {trade.exit_reason && (
        <View style={styles.exitReasonBadge}>
          <Text style={styles.exitReasonText}>
            {exitReasonLabel[trade.exit_reason] || trade.exit_reason}
          </Text>
        </View>
      )}

      {/* Model attribution */}
      <View style={styles.attribution}>
        <Text style={styles.attrLabel}>
          Signal: {fmt.sig(trade.ensemble_signal)}
        </Text>
        {trade.shares && (
          <Text style={styles.attrLabel}>
            Shares: {fmt.qty(trade.shares)}
          </Text>
        )}
      </View>
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
  title: {
    color: colors.text,
    fontSize: fontSize.xl,
    fontWeight: '700',
    marginBottom: spacing.md,
  },
  statsRow: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  stat: {
    flex: 1,
    backgroundColor: colors.card,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.cardBorder,
    padding: spacing.sm,
    alignItems: 'center',
  },
  statLabel: {
    color: colors.textMuted,
    fontSize: fontSize.xs,
    marginBottom: 2,
  },
  statValue: {
    color: colors.text,
    fontSize: fontSize.lg,
    fontWeight: '700',
  },
  tradeCard: {
    backgroundColor: colors.card,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.cardBorder,
    padding: spacing.md,
    marginBottom: spacing.sm,
  },
  tradeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  tradeTickerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  tradeTicker: {
    color: colors.text,
    fontSize: fontSize.md,
    fontWeight: '700',
  },
  tradeSide: {
    fontSize: fontSize.xs,
    fontWeight: '700',
  },
  tradePnl: {
    fontSize: fontSize.md,
    fontWeight: '700',
  },
  tradeOpen: {
    color: colors.blue,
    fontSize: fontSize.xs,
    fontWeight: '700',
  },
  tradeDetails: {
    marginBottom: spacing.xs,
  },
  tradeDetail: {
    color: colors.textSecondary,
    fontSize: fontSize.xs,
  },
  exitReasonBadge: {
    backgroundColor: colors.surface,
    borderRadius: 4,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    alignSelf: 'flex-start',
    marginBottom: spacing.xs,
  },
  exitReasonText: {
    color: colors.textSecondary,
    fontSize: fontSize.xs,
    fontWeight: '600',
  },
  attribution: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  attrLabel: {
    color: colors.textMuted,
    fontSize: fontSize.xs,
  },
  emptyText: {
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: 40,
    fontSize: fontSize.md,
  },
});
