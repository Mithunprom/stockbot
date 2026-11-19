export const SEARCH = {
  "timestamp": "2026-11-13T07:49:34.639759+00:00",
  "windows": {
    "tune": [
      "2022-01-01",
      "2024-12-31"
    ],
    "val": [
      "2025-01-01",
      "2025-12-31"
    ],
    "holdout": [
      "2026-01-01",
      "2026-11-01"
    ]
  },
  "universe": [
    "AAPL",
    "AMD",
    "AMZN",
    "AVGO",
    "GOOGL",
    "HD",
    "JPM",
    "MA",
    "META",
    "MSFT",
    "NVDA",
    "PG",
    "QQQ",
    "TSLA",
    "UNH",
    "V",
    "XOM"
  ],
  "n_tickers": 17,
  "generations": 3,
  "winner": {
    "spec": {
      "name": "alpaca_five",
      "topology": "alpaca_five",
      "source": "huygiatrng/AlpacaTradingAgent",
      "analysts": [
        "technical",
        "sentiment",
        "news",
        "fundamental",
        "macro"
      ],
      "weights": {
        "technical": 0.22,
        "sentiment": 0.18,
        "news": 0.18,
        "fundamental": 0.2,
        "macro": 0.22
      },
      "debate_weight": 0.5,
      "risk_mode": "balanced",
      "entry_threshold": 0.2,
      "stop_atr": 1.8,
      "tp_atr": 2.5,
      "holding_days": 5,
      "max_positions": 5,
      "size_pct": 0.08,
      "long_only": true,
      "trend_filter": true,
      "hmm_filter": false,
      "heat_ceiling": 0.8,
      "cost_bps": 5.0
    },
    "tune": {
      "label": "alpaca_five",
      "start": "2022-01-01",
      "end": "2024-12-31",
      "total_return_pct": -1.68,
      "sharpe": -0.128,
      "sortino": -0.138,
      "max_dd_pct": 5.91,
      "profit_factor": 0.944,
      "win_rate": 0.465,
      "n_trades": 301,
      "avg_hold_days": 4.1,
      "avg_win_pct": 2.72,
      "avg_loss_pct": -2.51,
      "calmar": -0.095,
      "score": -0.421,
      "n_days": 753,
      "halted": false,
      "exit_reasons": {
        "max_hold": 182,
        "stop_loss": 80,
        "take_profit": 38,
        "eod_liquidate": 1
      }
    },
    "validation": {
      "label": "alpaca_five",
      "start": "2025-01-01",
      "end": "2025-12-31",
      "total_return_pct": 1.13,
      "sharpe": 0.359,
      "sortino": 0.456,
      "max_dd_pct": 2.25,
      "profit_factor": 1.12,
      "win_rate": 0.475,
      "n_trades": 99,
      "avg_hold_days": 4.1,
      "avg_win_pct": 2.93,
      "avg_loss_pct": -2.38,
      "calmar": 0.505,
      "score": 0.157,
      "n_days": 250,
      "halted": false,
      "exit_reasons": {
        "max_hold": 59,
        "stop_loss": 25,
        "take_profit": 13,
        "eod_liquidate": 2
      }
    },
    "holdout": {
      "label": "alpaca_five",
      "start": "2026-01-01",
      "end": "2026-11-01",
      "total_return_pct": -0.81,
      "sharpe": -0.305,
      "sortino": -0.47,
      "max_dd_pct": 3.39,
      "profit_factor": 0.929,
      "win_rate": 0.461,
      "n_trades": 89,
      "avg_hold_days": 3.8,
      "avg_win_pct": 3.28,
      "avg_loss_pct": -3.11,
      "calmar": -0.289,
      "score": -0.613,
      "n_days": 209,
      "halted": false,
      "exit_reasons": {
        "max_hold": 41,
        "stop_loss": 30,
        "take_profit": 13,
        "eod_liquidate": 5
      }
    }
  },
  "buy_hold_validation": {
    "label": "buy_hold_equal",
    "start": "2025-01-01",
    "end": "2025-12-31",
    "total_return_pct": 19.6,
    "sharpe": 0.927,
    "sortino": 1.287,
    "max_dd_pct": 20.7,
    "profit_factor": 6.59,
    "win_rate": 0.824,
    "n_trades": 17,
    "avg_hold_days": 249.0,
    "avg_win_pct": 26.46,
    "avg_loss_pct": -18.74,
    "calmar": 0.954,
    "score": -3.712260903319449,
    "n_days": 250,
    "halted": false,
    "exit_reasons": {
      "eod_liquidate": 17
    }
  },
  "validation_leaderboard": [
    {
      "name": "alpaca_five",
      "topology": "alpaca_five",
      "source": "huygiatrng/AlpacaTradingAgent",
      "risk_mode": "balanced",
      "size_pct": 0.08,
      "heat_ceiling": 0.8,
      "entry_threshold": 0.2,
      "stop_atr": 1.8,
      "holding_days": 5,
      "trend_filter": true,
      "hmm_filter": false,
      "metrics": {
        "label": "alpaca_five",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "total_return_pct": 1.13,
        "sharpe": 0.359,
        "sortino": 0.456,
        "max_dd_pct": 2.25,
        "profit_factor": 1.12,
        "win_rate": 0.475,
        "n_trades": 99,
        "avg_hold_days": 4.1,
        "avg_win_pct": 2.93,
        "avg_loss_pct": -2.38,
        "calmar": 0.505,
        "score": 0.157,
        "n_days": 250,
        "halted": false,
        "exit_reasons": {
          "max_hold": 59,
          "stop_loss": 25,
          "take_profit": 13,
          "eod_liquidate": 2
        }
      }
    },
    {
      "name": "alpaca_five~thr~797069",
      "topology": "alpaca_five",
      "source": "huygiatrng/AlpacaTradingAgent",
      "risk_mode": "balanced",
      "size_pct": 0.08,
      "heat_ceiling": 0.8,
      "entry_threshold": 0.17518100400720238,
      "stop_atr": 1.8,
      "holding_days": 5,
      "trend_filter": true,
      "hmm_filter": false,
      "metrics": {
        "label": "alpaca_five~thr~797069",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "total_return_pct": 0.61,
        "sharpe": 0.177,
        "sortino": 0.227,
        "max_dd_pct": 2.38,
        "profit_factor": 1.042,
        "win_rate": 0.465,
        "n_trades": 142,
        "avg_hold_days": 4.0,
        "avg_win_pct": 3.0,
        "avg_loss_pct": -2.57,
        "calmar": 0.259,
        "score": -0.077,
        "n_days": 250,
        "halted": false,
        "exit_reasons": {
          "max_hold": 81,
          "stop_loss": 38,
          "take_profit": 20,
          "eod_liquidate": 3
        }
      }
    },
    {
      "name": "x_alpaca_five_830047",
      "topology": "alpaca_five",
      "source": "huygiatrng/AlpacaTradingAgent",
      "risk_mode": "committee",
      "size_pct": 0.08,
      "heat_ceiling": 0.8,
      "entry_threshold": 0.24,
      "stop_atr": 1.08,
      "holding_days": 10,
      "trend_filter": true,
      "hmm_filter": true,
      "metrics": {
        "label": "x_alpaca_five_830047",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "total_return_pct": -0.97,
        "sharpe": -0.734,
        "sortino": -0.508,
        "max_dd_pct": 2.47,
        "profit_factor": 0.651,
        "win_rate": 0.3,
        "n_trades": 20,
        "avg_hold_days": 4.7,
        "avg_win_pct": 4.2,
        "avg_loss_pct": -2.58,
        "calmar": -0.395,
        "score": -1.474,
        "n_days": 250,
        "halted": false,
        "exit_reasons": {
          "stop_loss": 12,
          "take_profit": 4,
          "max_hold": 4
        }
      }
    },
    {
      "name": "hybrid_defensive_search~w~180552",
      "topology": "hybrid",
      "source": "hybrid: small-size committee + trend/HMM filters",
      "risk_mode": "committee",
      "size_pct": 0.04,
      "heat_ceiling": 0.4,
      "entry_threshold": 0.24,
      "stop_atr": 1.08,
      "holding_days": 10,
      "trend_filter": true,
      "hmm_filter": true,
      "metrics": {
        "label": "hybrid_defensive_search~w~180552",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "total_return_pct": -0.82,
        "sharpe": -1.293,
        "sortino": -0.707,
        "max_dd_pct": 0.96,
        "profit_factor": 0.475,
        "win_rate": 0.167,
        "n_trades": 18,
        "avg_hold_days": 3.1,
        "avg_win_pct": 6.22,
        "avg_loss_pct": -2.71,
        "calmar": -0.865,
        "score": -2.37,
        "n_days": 250,
        "halted": false,
        "exit_reasons": {
          "stop_loss": 15,
          "take_profit": 3
        }
      }
    },
    {
      "name": "buy_hold_equal",
      "topology": "buy_hold",
      "source": "baseline",
      "risk_mode": "none",
      "size_pct": 0.0,
      "heat_ceiling": 0.8,
      "entry_threshold": 0.0,
      "stop_atr": 99.0,
      "holding_days": 10000,
      "trend_filter": false,
      "hmm_filter": false,
      "metrics": {
        "label": "buy_hold_equal",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "total_return_pct": 19.6,
        "sharpe": 0.927,
        "sortino": 1.287,
        "max_dd_pct": 20.7,
        "profit_factor": 6.59,
        "win_rate": 0.824,
        "n_trades": 17,
        "avg_hold_days": 249.0,
        "avg_win_pct": 26.46,
        "avg_loss_pct": -18.74,
        "calmar": 0.954,
        "score": -3.712260903319449,
        "n_days": 250,
        "halted": false,
        "exit_reasons": {
          "eod_liquidate": 17
        }
      }
    }
  ],
  "tune_top": [
    {
      "name": "x_alpaca_five_830047",
      "source": "huygiatrng/AlpacaTradingAgent",
      "metrics": {
        "label": "x_alpaca_five_830047",
        "start": "2022-01-01",
        "end": "2024-12-31",
        "total_return_pct": 0.77,
        "sharpe": 0.177,
        "sortino": 0.142,
        "max_dd_pct": 1.87,
        "profit_factor": 1.139,
        "win_rate": 0.388,
        "n_trades": 49,
        "avg_hold_days": 4.2,
        "avg_win_pct": 4.18,
        "avg_loss_pct": -2.54,
        "calmar": 0.138,
        "score": -0.192,
        "n_days": 753,
        "halted": false,
        "exit_reasons": {
          "stop_loss": 28,
          "take_profit": 12,
          "max_hold": 9
        }
      }
    },
    {
      "name": "alpaca_five~thr~797069",
      "source": "huygiatrng/AlpacaTradingAgent",
      "metrics": {
        "label": "alpaca_five~thr~797069",
        "start": "2022-01-01",
        "end": "2024-12-31",
        "total_return_pct": -0.97,
        "sharpe": -0.044,
        "sortino": -0.047,
        "max_dd_pct": 7.28,
        "profit_factor": 0.977,
        "win_rate": 0.506,
        "n_trades": 415,
        "avg_hold_days": 4.0,
        "avg_win_pct": 2.53,
        "avg_loss_pct": -2.67,
        "calmar": -0.044,
        "score": -0.241,
        "n_days": 753,
        "halted": false,
        "exit_reasons": {
          "max_hold": 247,
          "stop_loss": 111,
          "take_profit": 56,
          "eod_liquidate": 1
        }
      }
    },
    {
      "name": "alpaca_five",
      "source": "huygiatrng/AlpacaTradingAgent",
      "metrics": {
        "label": "alpaca_five",
        "start": "2022-01-01",
        "end": "2024-12-31",
        "total_return_pct": -1.68,
        "sharpe": -0.128,
        "sortino": -0.138,
        "max_dd_pct": 5.91,
        "profit_factor": 0.944,
        "win_rate": 0.465,
        "n_trades": 301,
        "avg_hold_days": 4.1,
        "avg_win_pct": 2.72,
        "avg_loss_pct": -2.51,
        "calmar": -0.095,
        "score": -0.421,
        "n_days": 753,
        "halted": false,
        "exit_reasons": {
          "max_hold": 182,
          "stop_loss": 80,
          "take_profit": 38,
          "eod_liquidate": 1
        }
      }
    },
    {
      "name": "alpaca_five~heat~533033",
      "source": "huygiatrng/AlpacaTradingAgent",
      "metrics": {
        "label": "alpaca_five~heat~533033",
        "start": "2022-01-01",
        "end": "2024-12-31",
        "total_return_pct": -1.68,
        "sharpe": -0.128,
        "sortino": -0.138,
        "max_dd_pct": 5.91,
        "profit_factor": 0.944,
        "win_rate": 0.465,
        "n_trades": 301,
        "avg_hold_days": 4.1,
        "avg_win_pct": 2.72,
        "avg_loss_pct": -2.51,
        "calmar": -0.095,
        "score": -0.421,
        "n_days": 753,
        "halted": false,
        "exit_reasons": {
          "max_hold": 182,
          "stop_loss": 80,
          "take_profit": 38,
          "eod_liquidate": 1
        }
      }
    },
    {
      "name": "hybrid_defensive_search~w~180552",
      "source": "hybrid: small-size committee + trend/HMM filters",
      "metrics": {
        "label": "hybrid_defensive_search~w~180552",
        "start": "2022-01-01",
        "end": "2024-12-31",
        "total_return_pct": -0.0,
        "sharpe": 0.003,
        "sortino": 0.002,
        "max_dd_pct": 1.37,
        "profit_factor": 0.999,
        "win_rate": 0.365,
        "n_trades": 52,
        "avg_hold_days": 3.7,
        "avg_win_pct": 4.43,
        "avg_loss_pct": -2.83,
        "calmar": -0.001,
        "score": -0.466,
        "n_days": 753,
        "halted": false,
        "exit_reasons": {
          "stop_loss": 33,
          "take_profit": 12,
          "max_hold": 7
        }
      }
    },
    {
      "name": "alpaca_five~stop~826159",
      "source": "huygiatrng/AlpacaTradingAgent",
      "metrics": {
        "label": "alpaca_five~stop~826159",
        "start": "2022-01-01",
        "end": "2024-12-31",
        "total_return_pct": -4.12,
        "sharpe": -0.279,
        "sortino": -0.305,
        "max_dd_pct": 7.82,
        "profit_factor": 0.9,
        "win_rate": 0.515,
        "n_trades": 400,
        "avg_hold_days": 4.2,
        "avg_win_pct": 2.47,
        "avg_loss_pct": -2.84,
        "calmar": -0.176,
        "score": -0.491,
        "n_days": 753,
        "halted": false,
        "exit_reasons": {
          "max_hold": 253,
          "stop_loss": 94,
          "take_profit": 52,
          "eod_liquidate": 1
        }
      }
    },
    {
      "name": "primo_linear",
      "source": "ivebotunac/PrimoAgent",
      "metrics": {
        "label": "primo_linear",
        "start": "2022-01-01",
        "end": "2024-12-31",
        "total_return_pct": -2.55,
        "sharpe": -0.142,
        "sortino": -0.161,
        "max_dd_pct": 8.09,
        "profit_factor": 0.949,
        "win_rate": 0.431,
        "n_trades": 367,
        "avg_hold_days": 4.8,
        "avg_win_pct": 3.26,
        "avg_loss_pct": -2.65,
        "calmar": -0.105,
        "score": -0.535,
        "n_days": 753,
        "halted": false,
        "exit_reasons": {
          "stop_loss": 175,
          "max_hold": 107,
          "take_profit": 79,
          "drawdown_halt": 6
        }
      }
    },
    {
      "name": "primo_linear~trend~150788",
      "source": "ivebotunac/PrimoAgent",
      "metrics": {
        "label": "primo_linear~trend~150788",
        "start": "2022-01-01",
        "end": "2024-12-31",
        "total_return_pct": -2.55,
        "sharpe": -0.142,
        "sortino": -0.161,
        "max_dd_pct": 8.09,
        "profit_factor": 0.949,
        "win_rate": 0.431,
        "n_trades": 367,
        "avg_hold_days": 4.8,
        "avg_win_pct": 3.26,
        "avg_loss_pct": -2.65,
        "calmar": -0.105,
        "score": -0.535,
        "n_days": 753,
        "halted": false,
        "exit_reasons": {
          "stop_loss": 175,
          "max_hold": 107,
          "take_profit": 79,
          "drawdown_halt": 6
        }
      }
    }
  ],
  "history": [
    {
      "gen": 0,
      "best": "alpaca_five",
      "best_score": -0.421,
      "n": 14
    },
    {
      "gen": 1,
      "best": "alpaca_five~thr~797069",
      "best_score": -0.241,
      "n": 12
    },
    {
      "gen": 2,
      "best": "x_alpaca_five_830047",
      "best_score": -0.192,
      "n": 12
    },
    {
      "gen": 3,
      "best": "x_alpaca_five_830047",
      "best_score": -0.192,
      "n": 12
    }
  ],
  "ready_for_paper": false,
  "escalations": [
    {
      "level": "INFO",
      "trigger": "Winner does not yet meet paper-promotion gates (Sharpe\u22650.8 both legs, DD\u226410% val)",
      "recommendation": "Leave live.yaml untouched; review staging proposal only"
    }
  ]
};

export const SEEDS = [
  {
    "id": "primo_linear",
    "label": "Primo linear",
    "source": "ivebotunac/PrimoAgent",
    "blurb": "Data \u2192 technical \u2192 7-feature news NLP \u2192 portfolio manager. Lightest token path.",
    "analysts": [
      "technical",
      "news"
    ],
    "weights": {
      "technical": 0.6,
      "news": 0.4
    },
    "debate_weight": 0.0,
    "risk_mode": "balanced",
    "entry_threshold": 0.18,
    "stop_atr": 1.4,
    "tp_atr": 2.4,
    "holding_days": 8,
    "size_pct": 0.1,
    "heat_ceiling": 0.8,
    "trend_filter": false,
    "hmm_filter": false,
    "debate": false,
    "committee": false
  },
  {
    "id": "trading_agents",
    "label": "TradingAgents",
    "source": "TauricResearch/TradingAgents",
    "blurb": "Four analysts, Bull/Bear debate, trader, then risk.",
    "analysts": [
      "technical",
      "fundamental",
      "sentiment",
      "news"
    ],
    "weights": {
      "technical": 0.3,
      "fundamental": 0.25,
      "sentiment": 0.2,
      "news": 0.25
    },
    "debate_weight": 0.6,
    "risk_mode": "balanced",
    "entry_threshold": 0.22,
    "stop_atr": 1.5,
    "tp_atr": 2.8,
    "holding_days": 10,
    "size_pct": 0.1,
    "heat_ceiling": 0.8,
    "trend_filter": true,
    "hmm_filter": false,
    "debate": true,
    "committee": false
  },
  {
    "id": "nexus_trade",
    "label": "NexusTrade",
    "source": "NastyRunner13/StockTradingAgents",
    "blurb": "Research judge plus a five-role risk committee (median CRO).",
    "analysts": [
      "technical",
      "fundamental",
      "sentiment",
      "news"
    ],
    "weights": {
      "technical": 0.28,
      "fundamental": 0.22,
      "sentiment": 0.25,
      "news": 0.25
    },
    "debate_weight": 0.7,
    "risk_mode": "committee",
    "entry_threshold": 0.25,
    "stop_atr": 1.6,
    "tp_atr": 3.0,
    "holding_days": 12,
    "size_pct": 0.12,
    "heat_ceiling": 0.8,
    "trend_filter": false,
    "hmm_filter": true,
    "debate": true,
    "committee": true
  },
  {
    "id": "quant_swing",
    "label": "Quant swing",
    "source": "kasyap1234/stock_bot_langgraph",
    "blurb": "Indicators, HMM regime, sequence predictor, swing stops and targets.",
    "analysts": [
      "technical",
      "quant",
      "macro"
    ],
    "weights": {
      "technical": 0.35,
      "quant": 0.45,
      "macro": 0.2
    },
    "debate_weight": 0.2,
    "risk_mode": "balanced",
    "entry_threshold": 0.16,
    "stop_atr": 1.2,
    "tp_atr": 2.2,
    "holding_days": 7,
    "size_pct": 0.1,
    "heat_ceiling": 0.8,
    "trend_filter": true,
    "hmm_filter": true,
    "debate": false,
    "committee": false
  },
  {
    "id": "alpaca_five",
    "label": "Alpaca five",
    "source": "huygiatrng/AlpacaTradingAgent",
    "blurb": "Five analysts including macro. Walk-forward winner on the composite \u2014 still not paper-ready.",
    "winner": true,
    "analysts": [
      "technical",
      "sentiment",
      "news",
      "fundamental",
      "macro"
    ],
    "weights": {
      "technical": 0.22,
      "sentiment": 0.18,
      "news": 0.18,
      "fundamental": 0.2,
      "macro": 0.22
    },
    "debate_weight": 0.5,
    "risk_mode": "balanced",
    "entry_threshold": 0.2,
    "stop_atr": 1.8,
    "tp_atr": 2.5,
    "holding_days": 5,
    "size_pct": 0.08,
    "heat_ceiling": 0.8,
    "trend_filter": true,
    "hmm_filter": false,
    "debate": true,
    "committee": false
  },
  {
    "id": "hybrid",
    "label": "Hybrid best-of",
    "source": "All five, fused",
    "blurb": "Every analyst, debate, committee, HMM and trend filters.",
    "analysts": [
      "technical",
      "fundamental",
      "sentiment",
      "news",
      "macro",
      "quant"
    ],
    "weights": {
      "technical": 0.22,
      "fundamental": 0.14,
      "sentiment": 0.12,
      "news": 0.12,
      "macro": 0.16,
      "quant": 0.24
    },
    "debate_weight": 0.55,
    "risk_mode": "committee",
    "entry_threshold": 0.2,
    "stop_atr": 1.4,
    "tp_atr": 2.6,
    "holding_days": 8,
    "size_pct": 0.1,
    "heat_ceiling": 0.75,
    "trend_filter": true,
    "hmm_filter": true,
    "debate": true,
    "committee": true
  }
];
