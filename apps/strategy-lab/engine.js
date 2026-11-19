/** Deterministic graph stand-ins — same math as src/graph/{analysts,debate}.py */

function clip(x, lo = -1, hi = 1) {
  if (!Number.isFinite(x)) return 0;
  return Math.min(hi, Math.max(lo, x));
}

function tanh(x, scale) {
  if (!Number.isFinite(x) || scale === 0) return 0;
  return Math.tanh(x / scale);
}

function regimeInt(value, fallback = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return fallback;
  return Number.parseInt(value, 10);
}

export function scoreTechnical(row) {
  const findings = [];
  let score = 0;
  const rsi = Number(row.rsi_14 ?? 50);
  if (rsi < 30) {
    score += 0.35;
    findings.push(`RSI oversold ${rsi.toFixed(1)}`);
  } else if (rsi < 40) score += 0.12;
  else if (rsi > 70) {
    score -= 0.35;
    findings.push(`RSI overbought ${rsi.toFixed(1)}`);
  } else if (rsi > 60) score -= 0.12;
  else score += (50 - rsi) / 200;

  const macd = Number(row.macd_hist ?? 0);
  score += tanh(macd, 0.5) * 0.25;
  if (macd > 0) findings.push("MACD hist > 0");

  score += clip(Number(row.ema_stack ?? 0)) * 0.25;

  const adx = Number(row.adx ?? 20);
  if (adx < 15) {
    score *= 0.65;
    findings.push(`weak ADX ${adx.toFixed(1)}`);
  } else if (adx > 25) findings.push(`trending ADX ${adx.toFixed(1)}`);

  const bb = Number(row.bb_pct ?? 0.5);
  if (bb < 0.1) score += 0.15;
  else if (bb > 0.9) score -= 0.15;

  const don = Number(row.donchian_pos ?? 0.5);
  if (don > 0.95) {
    score += 0.1;
    findings.push("Donchian breakout");
  } else if (don < 0.05) score -= 0.1;

  score += 0.08 * Math.sign(Number(row.ichimoku_cloud_dir ?? 0));
  const conf = Math.min(1, 0.35 + 0.02 * adx + 0.3 * Math.abs(clip(score)));
  return { name: "technical", score: clip(score), confidence: conf, findings };
}

export function scoreFundamental(row) {
  const findings = [];
  const distHigh = Number(row.dist_52w_high ?? 0);
  const dd = Number(row.drawdown_from_peak ?? 0);
  const vs200 = Number(row.close_vs_sma_200 ?? 0);
  const r21 = Number(row.ret_21 ?? 0);
  const rvol = Number(row.rvol_60 ?? 0.2);
  let score = tanh(vs200, 0.15) * 0.45 + tanh(r21, 0.1) * 0.25;
  if (distHigh < -0.25) {
    score -= 0.15;
    findings.push("far from 52w high");
  } else if (distHigh > -0.05) {
    score += 0.1;
    findings.push("near 52w high");
  }
  if (dd < -0.2) {
    score -= 0.2;
    findings.push(`drawdown ${(dd * 100).toFixed(1)}%`);
  }
  if (rvol > 0.55) {
    score *= 0.7;
    findings.push("high 60d vol");
  }
  const conf = Math.min(1, 0.45 + 0.25 * Math.min(1, Math.abs(vs200) / 0.12));
  return { name: "fundamental", score: clip(score), confidence: conf, findings };
}

export function scoreSentiment(row) {
  const findings = [];
  const r5 = Number(row.ret_5 ?? 0);
  const volZ = Number(row.vol_z ?? 0);
  const mfi = Number(row.mfi ?? 50);
  const cmf = Number(row.cmf ?? 0);
  const force = Number(row.force_index_z ?? 0);
  let score = tanh(r5, 0.06) * 0.4;
  score += tanh(volZ, 2) * Math.sign(r5 + 1e-9) * 0.2;
  score += tanh((mfi - 50) / 50, 1) * 0.2;
  score += clip(cmf) * 0.15;
  score += tanh(force, 2) * 0.1;
  if (volZ > 1.5) findings.push(`volume z=${volZ.toFixed(1)}`);
  if (mfi > 80) findings.push("MFI hot");
  else if (mfi < 20) findings.push("MFI washed out");
  const conf = Math.min(1, 0.4 + (0.15 * Math.min(2, Math.abs(volZ))) / 2);
  return { name: "sentiment", score: clip(score), confidence: conf, findings };
}

export function scoreNews(row) {
  const gap = Number(row.gap ?? 0);
  const rng = Number(row.high_low_range ?? 0);
  const volZ = Number(row.vol_z ?? 0);
  const atr = Number(row.atr_pct ?? 0.015);
  const r1 = Number(row.ret_1 ?? 0);
  const r5 = Number(row.ret_5 ?? 0);
  const vs20 = Number(row.close_vs_sma_20 ?? 0);
  const newsRelevance = clip(Math.abs(volZ) / 3);
  const sentiment = clip(gap / 0.03);
  const priceImpact = clip((rng / Math.max(atr, 1e-4) - 1) / 2);
  const trendDirection = clip(vs20 / 0.05);
  const earningsImpact = Math.abs(gap) > 0.02 ? clip(gap / 0.04) : 0;
  const investorConfidence = clip(r5 / 0.08);
  const riskProfileChange = clip(-Math.abs(volZ) / 4);
  const feats = [
    Math.sign(r1) * newsRelevance,
    sentiment,
    Math.sign(r1) * Math.abs(priceImpact),
    trendDirection,
    earningsImpact,
    investorConfidence,
    riskProfileChange,
  ];
  const score = feats.reduce((a, b) => a + b, 0) / feats.length;
  const conf = Math.min(1, 0.35 + 0.25 * Math.min(1, Math.abs(volZ) / 2 + Math.abs(gap) / 0.03));
  const findings = [];
  if (Math.abs(gap) > 0.02) findings.push(`gap ${(gap * 100).toFixed(2)}%`);
  if (volZ > 2) findings.push("news-like volume");
  return { name: "news", score: clip(score), confidence: conf, findings };
}

export function scoreMacro(row) {
  const spyVs = Number(row.spy_vs_sma200 ?? 0);
  const spyVol = Number(row.spy_rvol_20 ?? 0.15);
  const rs = Number(row.rs_spy_21 ?? 0);
  const regime = regimeInt(row.hmm_regime, 1);
  let score = tanh(spyVs, 0.08) * 0.55 + tanh(rs, 0.08) * 0.3;
  if (spyVol > 0.28) score *= 0.6;
  if (regime === 2) score *= 0.5;
  const findings = [spyVs > 0 ? "SPY above 200d" : "SPY below 200d"];
  const conf = Math.abs(spyVs) > 0.02 ? 0.5 : 0.35;
  return { name: "macro", score: clip(score), confidence: conf, findings };
}

export function scoreQuant(row) {
  const pred = Number(row.seq_pred ?? 0);
  const regime = regimeInt(row.hmm_regime, 1);
  const atr = Number(row.atr_pct ?? 0.015);
  const don = Number(row.donchian_pos ?? 0.5);
  const rsi = Number(row.rsi_14 ?? 50);
  let score = tanh(pred, 0.03) * 0.55 + (don - 0.5) * 0.3 + (50 - rsi) / 200;
  if (regime === 2) score *= 0.45;
  else if (regime === 0) score *= 1.1;
  const findings = [`seq_pred=${pred.toFixed(3)}`, `regime=${regime}`, `atr=${(atr * 100).toFixed(2)}%`];
  const conf = Math.min(1, 0.4 + 8 * Math.abs(pred));
  return { name: "quant", score: clip(score), confidence: conf, findings };
}

const ANALYSTS = {
  technical: scoreTechnical,
  fundamental: scoreFundamental,
  sentiment: scoreSentiment,
  news: scoreNews,
  macro: scoreMacro,
  quant: scoreQuant,
};

export function runDebate(reports, weights, debateWeight = 0.55) {
  let raw = 0;
  let wsum = 0;
  let bullAcc = 0;
  let bearAcc = 0;
  let bullW = 0;
  let bearW = 0;
  for (const [name, report] of Object.entries(reports)) {
    const w = Number(weights[name] ?? 0);
    if (w <= 0) continue;
    const contrib = w * report.score * Math.max(report.confidence, 0.15);
    raw += contrib;
    wsum += w;
    if (report.score >= 0) {
      bullAcc += contrib;
      bullW += w;
    } else {
      bearAcc += -contrib;
      bearW += w;
    }
  }
  const rawScore = raw / Math.max(wsum, 1e-9);
  const scores = Object.values(reports).map((r) => r.score);
  let agreement = 0.5;
  if (scores.length > 1) {
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance = scores.reduce((a, s) => a + (s - mean) ** 2, 0) / scores.length;
    agreement = Math.max(0, 1 - variance);
  }
  const bullScore = bullW ? bullAcc / bullW : 0;
  const bearScore = bearW ? bearAcc / bearW : 0;
  const margin = bullScore - bearScore;
  const verdict = margin > 0.12 ? "bullish" : margin < -0.12 ? "bearish" : "neutral";
  const judgeScore = clip(0.65 * margin + 0.35 * rawScore);
  const confidence = Math.min(1, 0.3 + 0.5 * agreement + 0.2 * Math.abs(margin));
  const ensemble = clip((1 - debateWeight) * rawScore + debateWeight * judgeScore);
  return {
    ensemble,
    rawScore,
    debate: { bullScore, bearScore, judgeVerdict: verdict, judgeScore, confidence, margin, agreement },
  };
}

export function traderPlan(ensemble, confidence, atrPct, spec) {
  const atr = Math.max(Number(atrPct) || 0.015, 0.005);
  const stopPct = Math.min(Math.max(atr * spec.stop_atr, 0.008), 0.12);
  const targetPct = Math.min(Math.max(atr * spec.tp_atr, 0.012), 0.25);
  let action = "FLAT";
  if (ensemble >= spec.entry_threshold && confidence >= 0.3) action = "LONG";
  let size = spec.size_pct * Math.min(1, Math.abs(ensemble) / Math.max(spec.entry_threshold, 1e-6));
  if (action === "FLAT") size = 0;
  if (spec.trend_filter && Number(spec._spy_vs ?? 0) < 0 && action === "LONG") {
    /* applied in risk / signal filters; keep trader honest here */
  }
  return { action, score: ensemble, confidence, stopPct, targetPct, sizePct: size, holdingDays: spec.holding_days };
}

export function riskCommittee(plan, spec, ctx) {
  const daily = Number(ctx.daily_pnl_pct ?? 0);
  const dd = Number(ctx.drawdown_pct ?? 0);
  const heat = Number(ctx.portfolio_heat ?? 0);
  const regime = regimeInt(ctx.hmm_regime, 1);
  const spyVs = Number(ctx.spy_vs_sma200 ?? 0);
  const vix = Number(ctx.vix_proxy ?? 0.15);
  const losses = Number(ctx.consecutive_losses ?? 0);

  if (daily <= -0.03) return { decision: "rejected", sizePct: 0, reason: "daily_loss_limit", stopPct: plan.stopPct, targetPct: plan.targetPct };
  if (dd <= -0.08) return { decision: "rejected", sizePct: 0, reason: "max_drawdown", stopPct: plan.stopPct, targetPct: plan.targetPct };
  if (plan.action === "FLAT") return { decision: "rejected", sizePct: 0, reason: "flat", stopPct: plan.stopPct, targetPct: plan.targetPct };
  if (heat >= spec.heat_ceiling) return { decision: "rejected", sizePct: 0, reason: "heat_ceiling", stopPct: plan.stopPct, targetPct: plan.targetPct };
  if (vix > 0.35) return { decision: "rejected", sizePct: 0, reason: "vol_spike", stopPct: plan.stopPct, targetPct: plan.targetPct };

  let agg = 1.25;
  let cons = 0.55;
  let neu = 1.0;
  if (regime === 2) {
    agg = 0.8;
    cons = 0.25;
    neu = 0.5;
  }
  if (spyVs < 0 && plan.action === "LONG") {
    agg *= 0.85;
    cons *= 0.4;
    neu *= 0.7;
  }
  if (losses >= 5) {
    agg *= 0.5;
    cons *= 0.5;
    neu *= 0.5;
  }

  let mult = neu;
  let vetoConf = 0.35;
  const mode = spec.risk_mode;
  if (mode === "aggressive") {
    mult = agg;
    vetoConf = 0.22;
  } else if (mode === "conservative") {
    mult = cons;
    vetoConf = 0.55;
    if (regime === 2 || spyVs < -0.03) {
      return { decision: "rejected", sizePct: 0, reason: "conservative_regime", stopPct: plan.stopPct, targetPct: plan.targetPct };
    }
  } else if (mode === "committee") {
    mult = [agg, cons, neu].sort((a, b) => a - b)[1];
    vetoConf = 0.38;
    if (cons < 0.3 && plan.confidence < 0.55) {
      return { decision: "rejected", sizePct: 0, reason: "cro_conservative_veto", stopPct: plan.stopPct, targetPct: plan.targetPct };
    }
  }

  if (plan.confidence < vetoConf) {
    return { decision: "rejected", sizePct: 0, reason: "low_confidence", stopPct: plan.stopPct, targetPct: plan.targetPct };
  }

  let size = Math.min(plan.sizePct * mult, 0.2);
  size = Math.min(size, Math.max(0, spec.heat_ceiling - heat));
  if (size < 0.01) {
    return { decision: "rejected", sizePct: 0, reason: "size_floor", stopPct: plan.stopPct, targetPct: plan.targetPct };
  }

  let stopPct = plan.stopPct;
  let targetPct = plan.targetPct;
  if (mode === "conservative") {
    stopPct *= 0.85;
    targetPct *= 0.85;
  } else if (mode === "aggressive") {
    stopPct *= 1.1;
    targetPct *= 1.2;
  }
  const decision = Math.abs(mult - 1) > 0.05 ? "modified" : "approved";
  return { decision, sizePct: size, reason: "", stopPct, targetPct, sizeMult: mult };
}

export function runGraph(row, spec, ctx = {}) {
  const reports = {};
  for (const name of spec.analysts) {
    if (ANALYSTS[name]) reports[name] = ANALYSTS[name](row);
  }
  const { ensemble, rawScore, debate } = runDebate(reports, spec.weights, spec.debate_weight);
  const spyVs = Number(row.spy_vs_sma200 ?? 0);
  const vix = Number(row.spy_rvol_20 ?? 0.15);
  const regime = regimeInt(row.hmm_regime, 1);
  let filtered = ensemble;
  let filterReason = "";
  if (spec.trend_filter && spyVs < 0) {
    filtered = Math.min(filtered, spec.entry_threshold - 0.001);
    filterReason = "trend_filter (SPY below 200d)";
  }
  if (spec.hmm_filter && regime === 2) {
    filtered = Math.min(filtered, spec.entry_threshold - 0.001);
    filterReason = "hmm_filter (high-vol regime)";
  }
  if (vix > 0.35) {
    filtered = Math.min(filtered, spec.entry_threshold - 0.001);
    filterReason = "vix proxy > 35%";
  }
  const plan = traderPlan(filtered, debate.confidence, row.atr_pct, spec);
  const risk = riskCommittee(plan, spec, {
    daily_pnl_pct: ctx.daily_pnl_pct ?? 0,
    drawdown_pct: ctx.drawdown_pct ?? 0,
    portfolio_heat: ctx.portfolio_heat ?? 0,
    hmm_regime: regime,
    spy_vs_sma200: spyVs,
    vix_proxy: vix,
    consecutive_losses: ctx.consecutive_losses ?? 0,
  });
  const action = risk.decision === "rejected" ? "FLAT" : plan.action;
  return { reports, rawScore, ensemble: filtered, debate, plan, risk, action, filterReason };
}

export const DEFAULT_ROW = {
  rsi_14: 32,
  macd_hist: 0.35,
  ema_stack: 0.6,
  adx: 32,
  bb_pct: 0.42,
  donchian_pos: 0.85,
  ichimoku_cloud_dir: 1,
  dist_52w_high: -0.08,
  drawdown_from_peak: -0.06,
  close_vs_sma_200: 0.08,
  ret_21: 0.07,
  rvol_60: 0.22,
  ret_5: 0.04,
  vol_z: 1.4,
  mfi: 62,
  cmf: 0.2,
  force_index_z: 0.4,
  gap: 0.01,
  high_low_range: 0.018,
  atr_pct: 0.016,
  ret_1: 0.012,
  close_vs_sma_20: 0.05,
  spy_vs_sma200: 0.06,
  spy_rvol_20: 0.16,
  rs_spy_21: 0.03,
  hmm_regime: 0,
  seq_pred: 0.03,
  close: 100,
};

export const SCENARIOS = {
  trend: {
    label: "Uptrend day",
    note: "SPY above its 200-day, RSI not stretched, moderate volume. Typical LONG candidate.",
    row: { ...DEFAULT_ROW },
    ctx: { daily_pnl_pct: 0, drawdown_pct: -0.01, portfolio_heat: 0.2, consecutive_losses: 0 },
  },
  overbought: {
    label: "Overbought squeeze",
    note: "RSI 78 and a blow-off range. Technicals flip bearish; most graphs stay FLAT.",
    row: {
      ...DEFAULT_ROW,
      rsi_14: 82,
      bb_pct: 0.98,
      donchian_pos: 0.99,
      ret_5: 0.11,
      vol_z: 2.8,
      macd_hist: -0.25,
      ema_stack: 0.05,
      mfi: 86,
      seq_pred: -0.02,
    },
    ctx: { daily_pnl_pct: 0, drawdown_pct: -0.01, portfolio_heat: 0.2, consecutive_losses: 0 },
  },
  riskoff: {
    label: "Risk-off tape",
    note: "SPY below 200-day, realized vol elevated, HMM high-vol. Trend/HMM filters and the committee veto.",
    row: { ...DEFAULT_ROW, spy_vs_sma200: -0.06, spy_rvol_20: 0.32, hmm_regime: 2, rsi_14: 44, seq_pred: -0.01 },
    ctx: { daily_pnl_pct: 0, drawdown_pct: -0.02, portfolio_heat: 0.3, consecutive_losses: 0 },
  },
  dailyHalt: {
    label: "Daily −3% halt",
    note: "Same features as the uptrend, but the book is already −3.2% on the day. Circuit breaker rejects.",
    row: { ...DEFAULT_ROW },
    ctx: { daily_pnl_pct: -0.032, drawdown_pct: -0.04, portfolio_heat: 0.4, consecutive_losses: 2 },
  },
};
