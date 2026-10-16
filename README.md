"""
Meme-coin trend-following bot (EMA crossover), long-only spot trading.

Features:
 - EMA crossover signal (fast/slow)
 - Backtest / Paper / Live modes
 - Stop-loss, optional trailing stop, optional take-profit
 - State persistence per-symbol
 - Safe live order checks

Install:  pip install ccxt pandas
Examples:
  python meme_bot.py --backtest
  python meme_bot.py --symbol PEPE/USD --timeframe 1h --trailing 0.03 --take 0.06
  python meme_bot.py --live --trailing 0.02 --take 0.04

Live keys (never hardcode them):
  export EXCHANGE_API_KEY="..."
  export EXCHANGE_SECRET="..."
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

STATE_FILE = "bot_state.json"
LOG_FILE = "trades.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Meme-coin EMA crossover bot")
    p.add_argument("--exchange", default="kraken", help="any ccxt exchange id")
    p.add_argument("--symbol", default="DOGE/USD", help="e.g. DOGE/USD, SHIB/USD, PEPE/USD")
    p.add_argument("--timeframe", default="15m", help="1m 5m 15m 30m 1h 4h 1d")
    p.add_argument("--fast", type=int, default=9, help="fast EMA length")
    p.add_argument("--slow", type=int, default=21, help="slow EMA length")
    p.add_argument("--cash", type=float, default=100.0, help="paper/backtest starting cash")
    p.add_argument("--risk", type=float, default=0.5, help="fraction of cash used per buy (0-1)")
    p.add_argument("--stop", type=float, default=0.05, help="stop-loss, 0.05 = 5%")
    p.add_argument("--fee", type=float, default=0.0026, help="fee per trade, 0.0026 = 0.26%")
    p.add_argument("--trailing", type=float, default=0.0, help="trailing stop fraction, e.g. 0.03 for 3%% (0 to disable)")
    p.add_argument("--take", type=float, default=0.0, help="take-profit fraction, e.g. 0.05 for 5%% (0 to disable)")

    group = p.add_mutually_exclusive_group()
    group.add_argument("--backtest", action="store_true", help="run a backtest only")
    group.add_argument("--live", action="store_true", help="place real orders")

    return p.parse_args()


# ---------- data + signal ----------

def fetch_df(ex, symbol, timeframe, limit=720):
    raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    if not raw:
        raise ValueError(f"No OHLCV data returned for {symbol} on {timeframe}.")
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol"])
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts", "open", "high", "low", "close", "vol"]).reset_index(drop=True)


def add_emas(df, fast, slow):
    if len(df) < 2:
        return df.copy()
    df = df.copy()
    df["fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df


def cross(prev, cur):
    """'buy' when fast EMA crosses above slow, 'sell' when it crosses below."""
    try:
        if prev.fast <= prev.slow and cur.fast > cur.slow:
            return "buy"
        if prev.fast >= prev.slow and cur.fast < cur.slow:
            return "sell"
    except Exception:
        return None
    return None


# ---------- backtest ----------

def backtest(df, a):
    df = add_emas(df.copy(), a.fast, a.slow)
    if len(df) <= a.slow + 1:
        raise SystemExit("Not enough candles for backtest; increase limit or choose a longer timeframe.")

    start = a.slow + 1
    cash, qty, entry = a.cash, 0.0, 0.0
    wins, trades = 0, 0
    max_price = 0.0
    stop_price = 0.0
    take_price = 0.0

    for i in range(start, len(df)):
        prev, row = df.iloc[i - 1], df.iloc[i]

        # update trailing max_price when in position
        if qty > 0:
            max_price = max(max_price, row.high)
            if a.trailing > 0:
                stop_price = max(stop_price, max_price * (1 - a.trailing))
            else:
                stop_price = entry * (1 - a.stop)
            if a.take > 0:
                take_price = entry * (1 + a.take)

            # check take-profit first (assume filled at take price)
            if a.take > 0 and row.high >= take_price:
                proceeds = qty * take_price * (1 - a.fee)
                wins += int(proceeds > qty * entry)
                cash += proceeds
                qty = 0.0
                trades += 1
                continue

            # check stop-loss / trailing stop (assume filled at stop_price)
            if row.low <= stop_price:
                px = stop_price
                cash += qty * px * (1 - a.fee)
                qty = 0.0
                trades += 1
                continue

        # regular crossover signals
        sig = cross(prev, row)
        if sig == "buy" and qty == 0:
            spend = cash * a.risk
            if spend <= 0:
                continue
            qty = spend * (1 - a.fee) / row.close
            cash -= spend
            entry = row.close
            max_price = entry
            stop_price = entry * (1 - a.stop)
            take_price = entry * (1 + a.take) if a.take > 0 else 0.0

        elif sig == "sell" and qty > 0:
            proceeds