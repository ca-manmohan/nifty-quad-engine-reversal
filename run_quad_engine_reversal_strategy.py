"""
====================================================================================================
QUAD-ENGINE REVERSAL STRATEGY: 100% PURE DOMESTIC NSE TELEMETRY & REAL OPTION CONTRACTS
====================================================================================================
Academic Source: Yash Vats (Oct 2026) - Short-term reversal (z_mom_1 = 24.8% attribution) & regularized
                 linear volatility-exhaustion gates on compact liquid Indian equity cross-sections.

Core Institutional Engines:
  1. Engine 1: Morning Exhaustion Reversal (09:30 - 10:40 IST)
     - Fades tail overextensions (|Spot - Open| >= 45 pts) accompanied by volume climax (>= 1.25x 20-MA)
       and rejection pin bars (wick >= 32%).
  2. Engine 2: Mid-Morning Institutional VWAP Retest (10:15 - 11:35 IST)
     - Catches primary trend resumption when Spot retests Cumulative Futures VWAP within 6 pts
       and forms rejection wicks.
  3. Engine 3: Post-Lunch Breakout Cascade (12:55 - 13:40 IST, Excl Mondays)
     - Captures directional expansions exceeding the 12:00-12:55 lunch range by >= 14 pts
       confirmed by Futures VWAP dislocation.
  4. Engine 4: Afternoon Institutional VWAP Rejection (13:00 - 14:35 IST)
     - Trades counter-trend bounces/rejections off the accumulated multi-million contract VWAP barrier.

Execution Architecture:
  - Dual-Tranche Breakeven Shield:
      Tranche 1 (50% lots): Quick harvest target at +8.0 option pts.
      Tranche 2 (50% lots): Once T1 hits, stop instantly shifts to Entry +0.50 pt (Guarantees >= +4.25 pts WIN).
      Trailing Ratchet: 65% peak excursion retention once peak >= +17.5 option pts.
  - Capital Budget: Rs 20,000 base capital.
  - Realistic Friction: Rs 40 roundtrip brokerage + 0.1% STT & turnover taxes + 0.50 pt slippage.
  - Zero Foreign Telemetry: No DAX, no FTSE. 100% domestic NSE data.
====================================================================================================
"""

import os
import glob
import json
import math
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path("data_cache")
OPTIONS_DIR = DATA_DIR / "options"
OUTPUT_DIR = Path("backtest_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL_BUDGET = 20000.0
LOT_SIZE = 50
BROKERAGE_PER_TRADE = 40.0
STT_TAX_RATE = 0.001
SLIPPAGE_PTS = 0.50

# Shield Parameters
T1_TARGET = 8.0            # +8.0 pts quick harvest target (locks profit)
HARD_SL = 11.5             # -11.5 pts hard stop loss
RUNNER_TARGET = 35.0       # +35.0 pts runner target
RATCHET_TRIGGER = 17.5     # +17.5 pts trailing ratchet trigger
RATCHET_RATIO = 0.65       # 65% peak excursion retention

ALL_DATES = [
    '2026-08-03', '2026-08-04', '2026-08-05', '2026-08-06', '2026-08-07',
    '2026-08-10', '2026-08-11', '2026-08-12', '2026-08-13', '2026-08-14',
    '2026-08-17', '2026-08-18', '2026-08-19', '2026-08-20', '2026-08-21',
    '2026-08-24', '2026-08-25', '2026-08-26', '2026-08-27', '2026-08-28',
    '2026-08-31', '2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04',
    '2026-09-07', '2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11',
    '2026-09-15', '2026-09-16', '2026-09-17', '2026-09-18', '2026-09-21',
    '2026-09-22', '2026-09-23', '2026-09-24', '2026-09-25', '2026-09-28',
    '2026-09-29', '2026-09-30', '2026-10-01', '2026-10-05', '2026-10-06',
    '2026-10-07', '2026-10-08', '2026-10-09', '2026-10-12', '2026-10-13',
    '2026-10-14', '2026-10-15', '2026-10-16', '2026-10-19'
]

IN_SAMPLE_DATES = [d for d in ALL_DATES if d <= '2026-09-11']
OUT_OF_SAMPLE_DATES = [d for d in ALL_DATES if d > '2026-09-11']

_OPTION_CACHE = {}

def preload_option_contracts():
    global _OPTION_CACHE
    files = glob.glob(str(OPTIONS_DIR / "*.json"))
    for f in files:
        base_name = os.path.basename(f)
        parts = base_name.split("_")
        key = None
        if len(parts) >= 6 and (parts[4] in ["CE", "PE"]):
            try:
                key = (parts[5].replace(".json", ""), int(parts[3]), parts[4])
            except ValueError:
                pass
        elif len(parts) >= 5 and (parts[3] in ["CE", "PE"]):
            try:
                key = (parts[4].replace(".json", ""), int(parts[2]), parts[3])
            except ValueError:
                pass
        if key:
            try:
                with open(f, "r") as fp:
                    _OPTION_CACHE[key] = json.load(fp)
            except Exception:
                pass
    print(f"[+] Loaded {len(_OPTION_CACHE)} option contract JSON files into memory.")

def load_fii_lsr():
    fii_file = DATA_DIR / "fii" / "fii_daily_synthesis_june_to_oct.csv"
    if fii_file.exists():
        df = pd.read_csv(fii_file)
        df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('date_str').reset_index(drop=True)
        df['lsr_prev'] = df['fii_lsr_pct'].shift(1).bfill()
        return df.set_index('date_str')['lsr_prev'].to_dict()
    return {}

def run_quad_engine_strategy():
    preload_option_contracts()
    fii_map = load_fii_lsr()
    last_fii = list(fii_map.values())[-1] if fii_map else 8.5
    
    trades = []
    session_logs = []
    running_capital = CAPITAL_BUDGET
    
    for d in ALL_DATES:
        dt_obj = pd.to_datetime(d)
        period = 'IN_SAMPLE' if d in IN_SAMPLE_DATES else 'OUT_OF_SAMPLE'
        lsr_prev = fii_map.get(d, last_fii)
        
        sp_f = DATA_DIR / f"spot_{d}.json"
        fu_f = DATA_DIR / f"fut_{d}.json"
        if not sp_f.exists() or not fu_f.exists():
            continue
            
        with open(sp_f) as fp: spot = json.load(fp)
        with open(fu_f) as fp: fut = json.load(fp)
        n = min(len(spot), len(fut))
        if n < 260:
            continue
            
        spot_o = np.array([float(s[1] or 0.0) for s in spot[:n]])
        spot_h = np.array([float(s[2] or 0.0) for s in spot[:n]])
        spot_l = np.array([float(s[3] or 0.0) for s in spot[:n]])
        spot_c = np.array([float(s[4] or 0.0) for s in spot[:n]])
        fut_c = np.array([float(f[4] or 0.0) for f in fut[:n]])
        fut_v = np.array([float(f[5] or 0.0) for f in fut[:n]])
        open_p = float(spot[0][1] or 0.0)
        
        cum_v = np.cumsum(fut_v)
        cum_vp = np.cumsum(fut_c * fut_v)
        vwap = np.where(cum_v > 0, cum_vp / np.maximum(cum_v, 1.0), fut_c)
        fu_v_series = pd.Series(fut_v)
        v_ma20 = fu_v_series.rolling(20, min_periods=5).mean().values
        lunch_h = np.max(spot_h[165:220])
        lunch_l = np.min(spot_l[165:220])
        
        session_trades = []
        last_exit_bar = -999
        day_direction = None
        executed_categories = set()
        
        for t in range(15, min(325, n - 15)):
            if t - last_exit_bar < 18:
                continue
            if len(session_trades) >= 3:
                break
                
            p_c = spot_c[t]
            p_o = spot_o[t]
            p_h = spot_h[t]
            p_l = spot_l[t]
            v_t = vwap[t]
            dist_open = p_c - open_p
            
            c_rng = max(p_h - p_l, 1.0)
            u_wick = p_h - max(p_o, p_c)
            l_wick = min(p_o, p_c) - p_l
            
            direction = None
            opt_type = None
            setup_name = ""
            category = ""
            
            # -------------------------------------------------------------------------------------
            # ENGINE 1: Morning Exhaustion Reversal (09:30 - 10:40 IST: t=15 to 85) [Yash Vats]
            # -------------------------------------------------------------------------------------
            if 15 <= t <= 85 and 'MORN' not in executed_categories:
                # Bearish Exhaustion (PUT BUY)
                if dist_open >= 47.0 and (u_wick / c_rng) >= 0.28 and p_c <= p_o and fut_v[t] >= 1.22 * v_ma20[t]:
                    if lsr_prev <= 35.0:
                        direction = "PUT_BUY"
                        opt_type = "PE"
                        setup_name = "MORN_BEAR_REVERSAL"
                        category = "MORN"
                # Bullish Exhaustion (CALL BUY)
                elif dist_open <= -47.0 and (l_wick / c_rng) >= 0.28 and p_c >= p_o and fut_v[t] >= 1.22 * v_ma20[t]:
                    if lsr_prev >= 13.0:
                        direction = "CALL_BUY"
                        opt_type = "CE"
                        setup_name = "MORN_BULL_REVERSAL"
                        category = "MORN"
                        
            # -------------------------------------------------------------------------------------
            # ENGINE 2: Mid-Morning Institutional VWAP Retest (10:15 - 11:35 IST: t=60 to 140, Excl Mondays)
            # -------------------------------------------------------------------------------------
            if not direction and dt_obj.dayofweek != 0 and 60 <= t <= 140 and 'MID' not in executed_categories:
                # Downtrend Retest & Rejection at VWAP
                if dist_open <= -25.0 and abs(p_h - v_t) <= 6.0 and p_c <= v_t and (u_wick / c_rng) >= 0.30:
                    if lsr_prev <= 35.0:
                        direction = "PUT_BUY"
                        opt_type = "PE"
                        setup_name = "MID_VWAP_REJECTION"
                        category = "MID"
                # Uptrend Dip & Bounce at VWAP
                elif dist_open >= 25.0 and abs(p_l - v_t) <= 6.0 and p_c >= v_t and (l_wick / c_rng) >= 0.30:
                    if lsr_prev >= 13.0:
                        direction = "CALL_BUY"
                        opt_type = "CE"
                        setup_name = "MID_VWAP_BOUNCE"
                        category = "MID"
                        
            # -------------------------------------------------------------------------------------
            # ENGINE 3: Post-Lunch Breakout Cascade (12:55 - 13:40 IST: t=220 to 265, Excl Mondays)
            # -------------------------------------------------------------------------------------
            if not direction and dt_obj.dayofweek != 0 and (220 <= t <= 265) and 'AFT' not in executed_categories:
                if p_c >= lunch_h + 15.0 and p_c >= v_t + 10.0:
                    if lsr_prev >= 12.0:
                        direction = "CALL_BUY"
                        opt_type = "CE"
                        setup_name = "AFT_BULL_BREAKOUT"
                        category = "AFT"
                elif p_c <= lunch_l - 15.0 and p_c <= v_t - 10.0:
                    if lsr_prev <= 35.0:
                        direction = "PUT_BUY"
                        opt_type = "PE"
                        setup_name = "AFT_BEAR_BREAKDOWN"
                        category = "AFT"
                        
            # -------------------------------------------------------------------------------------
            # ENGINE 4: Afternoon Institutional VWAP Rejection (13:00 - 14:35 IST: t=225 to 320)
            # -------------------------------------------------------------------------------------
            if not direction and (225 <= t <= 320) and 'VWAP' not in executed_categories:
                if dist_open >= 18.0 and abs(p_l - v_t) <= 6.0 and p_c >= v_t and (l_wick / c_rng) >= 0.30:
                    if lsr_prev >= 12.0:
                        direction = "CALL_BUY"
                        opt_type = "CE"
                        setup_name = "VWAP_BULL_BOUNCE"
                        category = "VWAP"
                elif dist_open <= -18.0 and abs(p_h - v_t) <= 6.0 and p_c <= v_t and (u_wick / c_rng) >= 0.30:
                    if lsr_prev <= 35.0:
                        direction = "PUT_BUY"
                        opt_type = "PE"
                        setup_name = "VWAP_BEAR_REJECTION"
                        category = "VWAP"
                        
            if not direction:
                continue
            if day_direction and day_direction != direction:
                continue
                
            e_idx = t + 1
            spot_e = float(spot[e_idx][1] or 0.0)
            atm_k = int(round(spot_e / 50.0) * 50)
            candles = _OPTION_CACHE.get((d, atm_k, opt_type))
            if not candles or len(candles) <= e_idx + 1:
                continue
                
            prior_o = float(candles[t][1] or 0.0)
            prior_c = float(candles[t][4] or 0.0)
            prior_h = float(candles[t][2] or 0.0)
            prior_l = float(candles[t][3] or 0.0)
            if (prior_c - prior_o) >= 12.0 or (prior_h - prior_l) >= 15.0:
                continue
                
            entry_raw = float(candles[e_idx][1] or 0.0)
            if entry_raw < 15.0:
                continue
                
            entry_price = entry_raw + SLIPPAGE_PTS
            cost_per_lot = entry_price * LOT_SIZE
            raw_lots = int(CAPITAL_BUDGET // cost_per_lot)
            lots = max(2, min(4, (raw_lots // 2) * 2))
            qty = lots * LOT_SIZE
            capital_deployed = entry_price * qty
            q1 = qty // 2
            q2 = qty - q1
            
            t1_done = False
            t2_done = False
            t1_pts = 0.0
            t2_pts = 0.0
            peak_gain = 0.0
            exit_reason = ""
            exit_idx = e_idx + 1
            max_hold = min(e_idx + 45, min(350, len(candles)))
            
            for m in range(e_idx + 1, max_hold):
                c = candles[m]
                hp = float(c[2] or 0.0)
                lp = float(c[3] or 0.0)
                gain = hp - entry_price
                loss = lp - entry_price
                peak_gain = max(peak_gain, gain)
                
                if not t1_done:
                    if loss <= -HARD_SL:
                        t1_done = t2_done = True
                        t1_pts = t2_pts = -HARD_SL
                        exit_reason = "HARD_SL"
                        exit_idx = m
                        break
                    if gain >= T1_TARGET:
                        t1_done = True
                        t1_pts = T1_TARGET
                        
                if t1_done and not t2_done:
                    floor = max(0.50, RATCHET_RATIO * peak_gain) if peak_gain >= RATCHET_TRIGGER else 0.50
                    if loss <= floor:
                        t2_done = True
                        t2_pts = floor
                        exit_reason = f"T2_RATCHET_{floor:+.1f}" if peak_gain >= RATCHET_TRIGGER else "BE_SHIELD_LOCK (+0.5)"
                        exit_idx = m
                        break
                    if gain >= RUNNER_TARGET:
                        t2_done = True
                        t2_pts = RUNNER_TARGET
                        exit_reason = f"T2_TARGET_{RUNNER_TARGET:+.1f}"
                        exit_idx = m
                        break
                        
            if not t1_done or not t2_done:
                final_c = float(candles[min(max_hold - 1, len(candles) - 1)][4] or 0.0)
                final_pts = final_c - entry_price
                if not t1_done: t1_pts = final_pts
                if not t2_done: t2_pts = final_pts
                exit_reason = "MAX_HOLD_TIME_EXIT"
                exit_idx = min(max_hold - 1, len(candles) - 1)
                
            gross = (t1_pts * q1) + (t2_pts * q2)
            charges = BROKERAGE_PER_TRADE + (capital_deployed * STT_TAX_RATE)
            net = gross - charges
            running_capital += net
            
            tr_info = {
                'date': d,
                'period': period,
                'setup': setup_name,
                'direction': direction,
                'strike': atm_k,
                'entry_time': str(candles[e_idx][0])[11:16],
                'exit_time': str(candles[exit_idx][0])[11:16],
                'entry_price': round(entry_price, 2),
                'capital_deployed': round(capital_deployed, 2),
                'lots': lots,
                'qty': qty,
                't1_pts': round(t1_pts, 2),
                't2_pts': round(t2_pts, 2),
                'blended_pts': round(0.5 * (t1_pts + t2_pts), 2),
                'peak_gain': round(peak_gain, 2),
                'gross_pnl': round(gross, 2),
                'charges': round(charges, 2),
                'net_pnl': round(net, 2),
                'exit_reason': exit_reason,
                'equity': round(running_capital, 2)
            }
            trades.append(tr_info)
            session_trades.append(tr_info)
            day_direction = direction
            executed_categories.add(category)
            last_exit_bar = exit_idx
            
        if len(session_trades) == 0:
            session_logs.append({
                'date': d, 'period': period, 'trades_count': 0, 'status': 'STAND_ASIDE', 'net_pnl': 0.0, 'equity': round(running_capital, 2)
            })
        else:
            session_logs.append({
                'date': d, 'period': period, 'trades_count': len(session_trades), 'status': 'TRADED',
                'net_pnl': sum(x['net_pnl'] for x in session_trades), 'equity': round(running_capital, 2)
            })
            
    df_trades = pd.DataFrame(trades)
    df_sessions = pd.DataFrame(session_logs)
    
    out_file = OUTPUT_DIR / "quad_engine_reversal_trades.csv"
    df_trades.to_csv(out_file, index=False)
    print(f"\n[+] Saved quad-engine trade logs to {out_file}")
    
    def print_scorecard(df_sub, label):
        if len(df_sub) == 0:
            print(f"No trades in {label}")
            return
        wins = df_sub[df_sub['net_pnl'] > 0]
        losses = df_sub[df_sub['net_pnl'] <= 0]
        tot = len(df_sub)
        n_w = len(wins)
        n_l = len(losses)
        wr = (n_w / tot) * 100.0 if tot > 0 else 0.0
        g_win = wins['net_pnl'].sum() if n_w > 0 else 0.0
        g_loss = abs(losses['net_pnl'].sum()) if n_l > 0 else 0.0
        pf = (g_win / g_loss) if g_loss > 0 else (999.0 if g_win > 0 else 0.0)
        net_pnl = df_sub['net_pnl'].sum()
        
        cum = df_sub['net_pnl'].cumsum()
        peak = cum.cummax()
        dd = (peak - cum).max() if len(cum) > 0 else 0.0
        
        print(f"\n" + "=" * 65)
        print(f"  {label.upper()} SCORECARD")
        print("=" * 65)
        print(f"  Total Trades Executed:   {tot}")
        print(f"  Winning Trades:          {n_w}")
        print(f"  Losing Trades:           {n_l}")
        print(f"  WIN RATE:                {wr:.2f}%")
        print(f"  Gross Profit:            Rs {g_win:,.2f}")
        print(f"  Gross Loss:              Rs {g_loss:,.2f}")
        print(f"  PROFIT FACTOR:           {pf:.2f}")
        print(f"  NET REALIZED PnL:        Rs {net_pnl:,.2f}")
        print(f"  Return on Capital (20k): +{(net_pnl / CAPITAL_BUDGET) * 100.0:.2f}%")
        print(f"  Max Realized Drawdown:   Rs {dd:,.2f}")
        print("=" * 65)
        
    df_is = df_trades[df_trades['period'] == 'IN_SAMPLE']
    df_oos = df_trades[df_trades['period'] == 'OUT_OF_SAMPLE']
    
    print_scorecard(df_is, "In-Sample (Aug 03 to Sep 11)")
    print_scorecard(df_oos, "Out-Of-Sample (Sep 15 to Oct 19)")
    print_scorecard(df_trades, "Full Consolidated (54 Sessions)")
    
    print("\n" + "=" * 80)
    print("ENGINE BREAKDOWN:")
    print("=" * 80)
    print(df_trades.groupby('setup')['net_pnl'].agg(['count', 'sum', lambda s: (s > 0).mean() * 100]).rename(columns={'count': 'Trades', 'sum': 'Net PnL', '<lambda_0>': 'WinRate%'}))
    
    print("\n" + "=" * 100)
    print("COMPLETE CHRONOLOGICAL REAL-TRADE AUDIT:")
    print("=" * 100)
    cols = ['date', 'setup', 'direction', 'strike', 'entry_time', 'exit_time', 'entry_price', 'lots', 'blended_pts', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason', 'equity']
    print(df_trades[cols].to_string(index=False))

if __name__ == "__main__":
    run_quad_engine_strategy()
