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
            proceeds = qty * row.close * (1 - a.fee)
            wins += int(proceeds > qty * entry)
            cash += proceeds
            qty = 0.0
            trades += 1

    final = cash + qty * df.close.iloc[-1]
    hold = a.cash * df.close.iloc[-1] / df.close.iloc[start]
    print(f"\nBacktest {a.symbol} {a.timeframe}  ({len(df) - start} candles)")
    print(f"  Start:        ${a.cash:,.2f}")
    print(f"  Bot final:    ${final:,.2f}  ({(final / a.cash - 1) * 100:+.1f}%)")
    print(f"  Buy & hold:   ${hold:,.2f}  ({(hold / a.cash - 1) * 100:+.1f}%)")
    print(f"  Closed trades: {trades}")
    print("  Note: past results don't predict future results.\n")


# ---------- portfolio / logging ----------

class Position:
    def __init__(self, cash, symbol=None):
        self.cash = float(cash)
        self.qty = 0.0
        self.entry = 0.0
        self.symbol = symbol
        # trailing/take bookkeeping
        self.max_price = 0.0
        self.stop_price = 0.0
        self.take_price = 0.0

    @classmethod
    def load(cls, cash, live, symbol):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    d = json.load(f)
            except (json.JSONDecodeError, OSError):
                return cls(cash, symbol)

            if d.get("live") == live and d.get("symbol") == symbol:
                p = cls(d.get("cash", cash), symbol)
                p.qty = float(d.get("qty", 0.0))
                p.entry = float(d.get("entry", 0.0))
                p.max_price = float(d.get("max_price", p.entry or 0.0))
                p.stop_price = float(d.get("stop_price", 0.0))
                p.take_price = float(d.get("take_price", 0.0))
                return p

        return cls(cash, symbol)

    def save(self, live, symbol):
        with open(STATE_FILE, "w") as f:
            json.dump({
                "cash": self.cash,
                "qty": self.qty,
                "entry": self.entry,
                "symbol": symbol,
                "live": live,
                "max_price": self.max_price,
                "stop_price": self.stop_price,
                "take_price": self.take_price
            }, f)


def log_trade(side, symbol, price, qty, reason, live):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_utc", "mode", "side", "symbol", "price", "qty", "reason"])
        w.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "LIVE" if live else "PAPER",
            side,
            symbol,
            price,
            qty,
            reason
        ])
    print(f"[{'LIVE' if live else 'PAPER'}] {side.upper()} {qty:.6f} {symbol} @ {price} ({reason})")


def buy(ex, pos, a, price, reason):
    if a.live:
        quote = a.symbol.split("/")[1].upper()
        balance = ex.fetch_balance()
        free = float(balance.get("free", {}).get(quote, 0.0))
        budget = free * a.risk
        if budget <= 0:
            print("Not enough quote balance to buy.")
            return
        amount = budget / price
        try:
            amount = float(ex.amount_to_precision(a.symbol, amount))
        except Exception:
            amount = float("{:.8f}".format(amount))
        if amount <= 0:
            print("Calculated buy size is zero or invalid.")
            return
        order = ex.create_market_buy_order(a.symbol, amount)
        avg = float(order.get("average") or price)
        filled = float(order.get("filled") or amount)
        pos.qty = filled
        pos.entry = avg
        pos.cash = 0.0
    else:
        spend = pos.cash * a.risk
        if spend <= 0:
            print("No cash available to buy.")
            return
        pos.qty = spend * (1 - a.fee) / price
        pos.cash -= spend
        pos.entry = price

    pos.max_price = pos.entry
    pos.stop_price = pos.entry * (1 - a.stop)
    pos.take_price = pos.entry * (1 + a.take) if a.take > 0 else 0.0
    log_trade("buy", a.symbol, pos.entry, pos.qty, reason, a.live)


def sell(ex, pos, a, price, reason):
    qty = pos.qty
    if qty <= 0:
        return

    if a.live:
        try:
            amount = float(ex.amount_to_precision(a.symbol, qty))
        except Exception:
            amount = qty
        order = ex.create_market_sell_order(a.symbol, amount)
        price = float(order.get("average") or price)
    else:
        pos.cash += qty * price * (1 - a.fee)

    pos.qty = 0.0
    pos.entry = 0.0
    pos.max_price = 0.0
    pos.stop_price = 0.0
    pos.take_price = 0.0
    log_trade("sell", a.symbol, price, qty, reason, a.live)


# ---------- main loop ----------

def make_exchange(a):
    try:
        cls = getattr(ccxt, a.exchange)
    except AttributeError:
        raise SystemExit(f"Exchange '{a.exchange}' not found in ccxt.")
    cfg = {"enableRateLimit": True}
    if a.live:
        key, secret = os.getenv("EXCHANGE_API_KEY"), os.getenv("EXCHANGE_SECRET")
        if not key or not secret:
            raise SystemExit("Set EXCHANGE_API_KEY and EXCHANGE_SECRET env vars for --live.")
        cfg.update(apiKey=key, secret=secret)
    return cls(cfg)


def validate_symbol(ex, symbol):
    if not getattr(ex, "markets", None):
        ex.load_markets()
    markets = ex.markets
    if symbol not in markets:
        raise SystemExit(f"Symbol {symbol} not found on {ex.name}.")
    return True


def run(ex, a):
    if a.live:
        print("LIVE MODE: this will place REAL orders with REAL money.")
        if input("Type YES to continue: ").strip() != "YES":
            raise SystemExit("Cancelled.")

    validate_symbol(ex, a.symbol)

    pos = Position.load(a.cash, a.live, a.symbol)
    last_ts = None
    print(f"Running {'LIVE' if a.live else 'PAPER'} | {a.symbol} {a.timeframe} | "
          f"EMA {a.fast}/{a.slow} | stop {a.stop:.0%} | trailing {a.trailing:.0%} | take {a.take:.0%} | Ctrl+C to stop")

    while True:
        try:
            df = fetch_df(ex, a.symbol, a.timeframe, limit=300)
            if len(df) < a.slow + 2:
                time.sleep(30)
                continue

            ticker = ex.fetch_ticker(a.symbol)
            price = float(ticker.get("last") or ticker.get("close") or df["close"].iloc[-1])

            # update trailing bookkeeping from last known prices when in position
            if pos.qty > 0:
                pos.max_price = max(pos.max_price, price)
                if a.trailing > 0:
                    pos.stop_price = max(pos.stop_price, pos.max_price * (1 - a.trailing))
                else:
                    pos.stop_price = pos.entry * (1 - a.stop)
                if a.take > 0 and pos.take_price == 0.0:
                    pos.take_price = pos.entry * (1 + a.take)

                # check take-profit first
                if a.take > 0 and price >= pos.take_price:
                    sell(ex, pos, a, pos.take_price, "take-profit")
                    pos.save(a.live, a.symbol)
                    time.sleep(1)
                    continue

                # then trailing/stop-loss
                if price <= pos.stop_price:
                    sell(ex, pos, a, pos.stop_price, "stop/trailing")
                    pos.save(a.live, a.symbol)
                    time.sleep(1)
                    continue

            # check crossover once per closed candle
            closed = add_emas(df.iloc[:-1].copy(), a.fast, a.slow)  # drop forming candle
            if len(closed) < 2:
                time.sleep(30)
                continue

            ts = closed["ts"].iloc[-1]
            if ts != last_ts:
                last_ts = ts
                sig = cross(closed.iloc[-2], closed.iloc[-1])
                if sig == "buy" and pos.qty == 0:
                    buy(ex, pos, a, price, "EMA cross up")
                    pos.save(a.live, a.symbol)
                elif sig == "sell" and pos.qty > 0:
                    sell(ex, pos, a, price, "EMA cross down")
                    pos.save(a.live, a.symbol)

        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            print("Network issue, retrying:", e)
        except ccxt.ExchangeError as e:
            print("Exchange error:", e)
        except ValueError as e:
            print("Data issue:", e)
        except KeyboardInterrupt:
            print("\nStopped. Position state saved in", STATE_FILE)
            raise

        time.sleep(30)


def main():
    a = parse_args()
    if a.fast >= a.slow:
        raise SystemExit("--fast must be smaller than --slow")
    if not 0 <= a.risk <= 1:
        raise SystemExit("--risk must be between 0 and 1")
    if not 0 <= a.stop < 1:
        raise SystemExit("--stop must be between 0 and 1")
    if not 0 <= a.fee < 1:
        raise SystemExit("--fee must be between 0 and 1")
    if not 0 <= a.trailing < 1:
        raise SystemExit("--trailing must be between 0 and 1")
    if not 0 <= a.take < 1:
        raise SystemExit("--take must be between 0 and 1")

    ex = make_exchange(a)

    if a.backtest:
        backtest(fetch_df(ex, a.symbol, a.timeframe, limit=720), a)
    else:
        try:
            run(ex, a)
        except KeyboardInterrupt:
            print("\nStopped. Position state saved in", STATE_FILE)


if __name__ == "__main__":
    main()
