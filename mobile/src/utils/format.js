// Formatting utilities (ported from web frontend)

export const fmt = {
  price: (v) => {
    if (v == null) return '--';
    if (Math.abs(v) >= 1000) return `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    return `$${v.toFixed(2)}`;
  },

  pct: (v) => {
    if (v == null) return '--';
    const sign = v >= 0 ? '+' : '';
    return `${sign}${(v * 100).toFixed(2)}%`;
  },

  pctRaw: (v) => {
    if (v == null) return '--';
    const sign = v >= 0 ? '+' : '';
    return `${sign}${v.toFixed(2)}%`;
  },

  sig: (v) => {
    if (v == null) return '--';
    const sign = v >= 0 ? '+' : '';
    return `${sign}${v.toFixed(4)}`;
  },

  qty: (v) => {
    if (v == null) return '--';
    if (v >= 1) return v.toFixed(2);
    return v.toFixed(6); // crypto fractional
  },

  dollar: (v) => {
    if (v == null) return '--';
    const sign = v >= 0 ? '+' : '';
    return `${sign}$${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  },

  bars: (held, max) => `${held}/${max} bars`,
};
