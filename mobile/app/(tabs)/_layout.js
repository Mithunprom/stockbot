import { Tabs } from 'expo-router';
import { Text } from 'react-native';
import { colors } from '../../src/utils/theme';

function TabIcon({ label, focused }) {
  const icons = {
    Dashboard: focused ? '\u25C9' : '\u25CB',
    Signals: focused ? '\u26A1' : '\u2301',
    Positions: focused ? '\u25A0' : '\u25A1',
    Trades: focused ? '\u2611' : '\u2610',
    Risk: focused ? '\u26A0' : '\u25B3',
  };
  return (
    <Text style={{ color: focused ? colors.blue : colors.textMuted, fontSize: 18 }}>
      {icons[label] || '\u25CF'}
    </Text>
  );
}

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: colors.card,
          borderTopColor: colors.cardBorder,
          height: 85,
          paddingBottom: 25,
          paddingTop: 8,
        },
        tabBarActiveTintColor: colors.blue,
        tabBarInactiveTintColor: colors.textMuted,
        tabBarLabelStyle: { fontSize: 10, fontWeight: '600' },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Dashboard',
          tabBarIcon: ({ focused }) => <TabIcon label="Dashboard" focused={focused} />,
        }}
      />
      <Tabs.Screen
        name="signals"
        options={{
          title: 'Signals',
          tabBarIcon: ({ focused }) => <TabIcon label="Signals" focused={focused} />,
        }}
      />
      <Tabs.Screen
        name="positions"
        options={{
          title: 'Positions',
          tabBarIcon: ({ focused }) => <TabIcon label="Positions" focused={focused} />,
        }}
      />
      <Tabs.Screen
        name="trades"
        options={{
          title: 'Trades',
          tabBarIcon: ({ focused }) => <TabIcon label="Trades" focused={focused} />,
        }}
      />
      <Tabs.Screen
        name="risk"
        options={{
          title: 'Risk',
          tabBarIcon: ({ focused }) => <TabIcon label="Risk" focused={focused} />,
        }}
      />
    </Tabs>
  );
}
