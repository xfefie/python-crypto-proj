

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================
# CONFIG
# ============================
START = "2017-01-01"
END = None
OUT_DIR = "output"
INITIAL = 1_000_000

# Benchmarks
TICK_SPY = "SPY"
TICK_QQQ = "QQQ"
TICK_BOND = "AGG"

# Crypto proxies (spot; Engine 2 simulates futures/opts on top)
TICK_BTC = "BTC-USD"
TICK_ALT = "ETH-USD"

# Portfolio blend
W_ENGINE1 = 0.70
W_ENGINE2 = 0.30

# Rebalance assumptions (for applying trading costs on Engine 2 changes)
REB_FREQ = "W-FRI"
FEE_BPS = 6
SLIPPAGE_BPS = 3

# Rates / financing (research proxies)
STABLE_APR = 0.05     # baseline stable carry
MARGIN_APR = 0.08     # cost of leverage (Engine 2)
FUNDING_APR = 0.00    # optional net perp funding benefit/cost (Engine 2)

# ----------------------------
# ENGINE 1: high-Sharpe carry model (synthetic placeholder)
# Replace later with real PnL series from: funding/basis/arbitrage/DeFi carry.
# ----------------------------
ENGINE1_CARRY_APR = 0.14         # expected annual return for carry engine (e.g., blended stable + funding/basis)
ENGINE1_DAILY_VOL = 0.0025       # ~0.25% daily vol target (low noise)
ENGINE1_CRASH_GUARD = True       # de-risk (clip) during extreme crypto stress

# ----------------------------
# ENGINE 2: directional allocator (target Sharpe ~0.8-1.0 in best-case regimes)
# ----------------------------
VOL_TARGET_ANN_E2 = 0.22
MAX_GROSS_LEV_E2 = 2.00
MIN_CASH_E2 = 0.10
DD_THROTTLE_E2 = 0.12

# Stress-only hedge (Engine 2): rolling put only during high vol regime
OPTION_MAT_DAYS = 21
PUT_STRIKE = 0.92
PUT_BUDGET_ANNUAL = 0.020
IV_MULT = 1.25
STRESS_VOL_Q = 0.80  # hedge on when BTC vol is above this rolling quantile


# ============================
# METRICS
# ============================
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def ulcer_index(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd_pct = (equity / peak - 1.0) * 100.0
    return float(np.sqrt((dd_pct.clip(upper=0) ** 2).mean()))

def daily_rate_from_apr(apr: float) -> float:
    return (1 + apr) ** (1 / 365.25) - 1

def sharpe(daily_rets: pd.Series, rf_daily: float = 0.0) -> float:
    ex = daily_rets - rf_daily
    vol = ex.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(ex.mean() / vol * np.sqrt(252))

def info_ratio(port: pd.Series, bench: pd.Series) -> float:
    p, b = port.align(bench, join="inner")
    active = p - b
    te = active.std()
    if te == 0 or np.isnan(te):
        return np.nan
    return float(active.mean() / te * np.sqrt(252))

def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)

def ann_vol(daily_rets: pd.Series) -> float:
    return float(daily_rets.std() * np.sqrt(252))


# ============================
# DATA
# ============================
def get_prices(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all")
    df.columns = [c.upper() for c in df.columns]
    return df


# ============================
# ENGINE 2: Synthetic options (BS put)
# ============================
def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

@dataclass
class RollingPutOverlay:
    maturity_days: int = OPTION_MAT_DAYS
    strike_pct: float = PUT_STRIKE
    iv_mult: float = IV_MULT
    annual_budget: float = PUT_BUDGET_ANNUAL

    def run(self, btc_prices: pd.Series, vol_series: pd.Series, stress_mask: pd.Series) -> pd.Series:
        """
        Returns daily PnL as fraction of equity.
        Only buys/holds puts when stress_mask == 1, otherwise no hedge.
        """
        btc = btc_prices.dropna()
        idx = btc.index

        stress_mask = stress_mask.reindex(idx).fillna(0.0).astype(float)
        vol_series = vol_series.reindex(idx).ffill()

        pnl = pd.Series(0.0, index=idx)

        current_strike = None
        expiry = None
        hedge_on = False

        roll_dates = idx[::self.maturity_days].intersection(idx)

        for i, dt in enumerate(idx):
            S = float(btc.loc[dt])
            hedge_on = (stress_mask.loc[dt] == 1.0)

            if (dt in roll_dates) and hedge_on:
                current_strike = self.strike_pct * S
                expiry_i = min(i + self.maturity_days, len(idx) - 1)
                expiry = idx[expiry_i]

                T = self.maturity_days / 365.25
                sigma = max(float(self.iv_mult * vol_series.loc[dt]), 0.10)
                prem = bs_put_price(S=S, K=current_strike, T=T, r=0.0, sigma=sigma)

                prem_frac = prem / S
                prem_frac = min(prem_frac, self.annual_budget * (self.maturity_days / 252.0))
                pnl.loc[dt] -= prem_frac

            if expiry is not None and dt == expiry and current_strike is not None and hedge_on:
                intrinsic = max(current_strike - S, 0.0)
                payoff_frac = intrinsic / S
                pnl.loc[dt] += payoff_frac

            # if hedge turns off, we effectively stop holding options (simplification)
            if not hedge_on:
                current_strike = None
                expiry = None

        return pnl


# ============================
# ENGINE 2: Weights (participation + conditional leverage)
# ============================
def realized_vol_annual(series: pd.Series, lb: int = 20) -> pd.Series:
    return series.rolling(lb).std() * np.sqrt(252)

def make_weights_engine2(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.dropna()
    btc = px["BTC"]
    alt = px["ALT"]

    btc_ret = btc.pct_change()

    btc_ma_fast = btc.rolling(50).mean()
    btc_ma_slow = btc.rolling(200).mean()
    trend_on = (btc_ma_fast > btc_ma_slow).astype(float)

    btc_mom = (btc / btc.shift(63) - 1.0)
    alt_mom = (alt / alt.shift(63) - 1.0)

    # Better participation: risk-on if trend OR momo positive
    risk_gate = ((trend_on == 1) | (btc_mom > 0)).astype(float)
    alt_gate = ((risk_gate == 1) & (alt_mom > 0)).astype(float)

    # Base exposures (before vol targeting)
    btc_base = 0.85 * risk_gate
    alt_base = 0.35 * alt_gate
    gross = btc_base + alt_base

    btc_vol = realized_vol_annual(btc_ret, 20).clip(lower=0.10)
    scale = (VOL_TARGET_ANN_E2 / btc_vol).clip(0.0, 3.0)
    gross_scaled = (gross * scale).clip(0.0, MAX_GROSS_LEV_E2)

    denom = (btc_base + alt_base).replace(0, np.nan)
    btc_exp = (btc_base / denom) * gross_scaled
    alt_exp = (alt_base / denom) * gross_scaled
    btc_exp = btc_exp.fillna(0.0)
    alt_exp = alt_exp.fillna(0.0)

    cash = 1.0 - (btc_exp + alt_exp)

    # enforce min cash
    need_cash = (MIN_CASH_E2 - cash).clip(lower=0.0)
    shrink = (1.0 - need_cash / (btc_exp + alt_exp + 1e-12)).clip(0.0, 1.0)
    btc_exp *= shrink
    alt_exp *= shrink
    cash = 1.0 - (btc_exp + alt_exp)

    return pd.DataFrame({"CASH": cash, "BTC": btc_exp, "ALT": alt_exp}, index=px.index)


def backtest_engine2(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    hedge_pnl_frac: pd.Series,
    stable_apr: float,
    margin_apr: float,
    funding_apr: float,
    reb_freq: str,
    fee_bps: float,
    slip_bps: float,
    initial: float
) -> pd.Series:
    px = prices.dropna()
    w = weights.reindex(px.index).ffill().fillna(0.0)
    hedge = hedge_pnl_frac.reindex(px.index).fillna(0.0)

    rf_d = daily_rate_from_apr(stable_apr)
    margin_d = daily_rate_from_apr(margin_apr)
    funding_d = daily_rate_from_apr(funding_apr)

    rebal_dates = px.resample(reb_freq).last().index.intersection(px.index)

    eq = pd.Series(index=px.index, dtype=float)
    eq.iloc[0] = initial

    prev_w = w.iloc[0].copy()
    peak = initial

    for i in range(1, len(px.index)):
        dt = px.index[i]
        prev = px.index[i - 1]

        r_btc = px.loc[dt, "BTC"] / px.loc[prev, "BTC"] - 1.0
        r_alt = px.loc[dt, "ALT"] / px.loc[prev, "ALT"] - 1.0

        btc_exp = float(prev_w["BTC"])
        alt_exp = float(prev_w["ALT"])
        cash_w = float(prev_w["CASH"])

        gross_exp = btc_exp + alt_exp
        borrowed = max(gross_exp - 1.0, 0.0)

        gross_r = (
            cash_w * rf_d
            + btc_exp * r_btc
            + alt_exp * r_alt
            - borrowed * margin_d
            + gross_exp * funding_d
        )

        eq.iloc[i] = eq.iloc[i - 1] * (1.0 + gross_r + float(hedge.loc[dt]))

        peak = max(peak, eq.iloc[i])
        dd = eq.iloc[i] / peak - 1.0
        if dd < -DD_THROTTLE_E2:
            prev_w["BTC"] *= 0.35
            prev_w["ALT"] *= 0.35
            prev_w["CASH"] = 1.0 - (prev_w["BTC"] + prev_w["ALT"])

        if dt in rebal_dates:
            new_w = w.loc[dt].copy()
            turnover = float((new_w - prev_w).abs().sum())
            cost = (fee_bps + slip_bps) / 10_000.0 * turnover
            eq.iloc[i] *= (1.0 - cost)
            prev_w = new_w

    return eq


# ============================
# ENGINE 1: High-Sharpe carry stream (synthetic placeholder)
# ============================
def engine1_carry_series(index: pd.DatetimeIndex, btc_returns: pd.Series | None = None) -> pd.Series:
    """
    Generates a synthetic daily return stream for Engine 1.
    Designed to be high Sharpe: steady positive carry + small noise.
    Optional "crash guard": reduce/clip drawdowns during extreme BTC stress.

    Replace this later with real PnL series from:
      - perp funding capture (delta-neutral)
      - basis trades
      - DeFi lending + hedged exposures
      - market-neutral stat arb
    """
    rng = np.random.default_rng(7)  # fixed seed for reproducibility

    mu_d = daily_rate_from_apr(ENGINE1_CARRY_APR)       # daily drift
    # gaussian noise (low vol)
    eps = rng.normal(loc=0.0, scale=ENGINE1_DAILY_VOL, size=len(index))
    r = pd.Series(mu_d + eps, index=index)

    if ENGINE1_CRASH_GUARD and btc_returns is not None:
        # if BTC has extreme down days, assume carry engine reduces risk and avoids tail losses
        btc_r = btc_returns.reindex(index).fillna(0.0)
        stress = (btc_r < btc_r.rolling(252).quantile(0.02)).astype(float)  # rare severe downside days
        # On stress days, clip negative carry outcomes (risk-off behavior)
        r = r.where(stress == 0.0, r.clip(lower=-0.002, upper=0.003))

    return r


def equity_from_returns(returns: pd.Series, initial: float) -> pd.Series:
    eq = (1.0 + returns.fillna(0.0)).cumprod() * initial
    eq.iloc[0] = initial
    return eq


# ============================
# Benchmarks
# ============================
def bench_equity(series: pd.Series, initial: float) -> pd.Series:
    px = series.dropna()
    rets = px.pct_change().fillna(0.0)
    return equity_from_returns(rets, initial)

def bench_6040(spy: pd.Series, bond: pd.Series, initial: float, w_spy=0.6, w_bond=0.4) -> pd.Series:
    df = pd.concat([spy, bond], axis=1).dropna()
    df.columns = ["SPY", "BOND"]
    rets = df.pct_change().fillna(0.0)
    port = w_spy * rets["SPY"] + w_bond * rets["BOND"]
    return equity_from_returns(port, initial)

def bench_btc_overlay(btc: pd.Series, initial: float, stable_apr: float, w_btc=0.30) -> pd.Series:
    px = btc.dropna()
    rets = px.pct_change().fillna(0.0)
    rf_d = daily_rate_from_apr(stable_apr)
    port = w_btc * rets + (1.0 - w_btc) * rf_d
    return equity_from_returns(port, initial)


# ============================
# MAIN: Build both engines + blend + evidence
# ============================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("WORKING DIRECTORY:", os.getcwd())

    tickers = [TICK_SPY, TICK_QQQ, TICK_BOND, TICK_BTC, TICK_ALT]
    prices = get_prices(tickers, START, END)
    print("Downloaded columns:", list(prices.columns))

    spy = prices[TICK_SPY].dropna()
    qqq = prices[TICK_QQQ].dropna()
    bond = prices[TICK_BOND].dropna()
    btc = prices[TICK_BTC].dropna()
    alt = prices[TICK_ALT].dropna()

    crypto_px = pd.concat([btc, alt], axis=1).dropna()
    crypto_px.columns = ["BTC", "ALT"]

    # Common date window for comparison
    start_dt, end_dt = crypto_px.index[0], crypto_px.index[-1]
    spy = spy.loc[start_dt:end_dt]
    qqq = qqq.loc[start_dt:end_dt]
    bond = bond.loc[start_dt:end_dt]
    btc = btc.loc[start_dt:end_dt]

    # =====================
    # Engine 2 build
    # =====================
    w2 = make_weights_engine2(crypto_px)
    btc_rets = crypto_px["BTC"].pct_change().fillna(0.0)
    btc_vol = realized_vol_annual(btc_rets, 20).clip(lower=0.10)

    # Stress mask for hedge: turn ON only when BTC vol is very high vs its own history
    vol_q = btc_vol.rolling(252).quantile(STRESS_VOL_Q)
    stress_mask = (btc_vol > vol_q).astype(float)

    hedge2 = RollingPutOverlay().run(crypto_px["BTC"], vol_series=btc_vol, stress_mask=stress_mask)

    eq2 = backtest_engine2(
        prices=crypto_px,
        weights=w2,
        hedge_pnl_frac=hedge2,
        stable_apr=STABLE_APR,
        margin_apr=MARGIN_APR,
        funding_apr=FUNDING_APR,
        reb_freq=REB_FREQ,
        fee_bps=FEE_BPS,
        slip_bps=SLIPPAGE_BPS,
        initial=INITIAL,
    )
    r2 = eq2.pct_change().fillna(0.0)

    # =====================
    # Engine 1 build (synthetic high-Sharpe carry)
    # =====================
    r1 = engine1_carry_series(index=eq2.index, btc_returns=btc_rets)
    eq1 = equity_from_returns(r1, INITIAL)

    # =====================
    # Blend: 70/30
    # =====================
    # Combine daily returns: portfolio return = w1*r1 + w2*r2
    port_r = (W_ENGINE1 * r1) + (W_ENGINE2 * r2)
    eq_port = equity_from_returns(port_r, INITIAL)

    # Benchmarks
    eq_spy = bench_equity(spy, INITIAL)
    eq_qqq = bench_equity(qqq, INITIAL)
    eq_6040 = bench_6040(spy, bond, INITIAL)
    eq_overlay = bench_btc_overlay(btc, INITIAL, stable_apr=STABLE_APR, w_btc=0.30)

    curves = pd.concat(
        [
            eq_port.rename("TwoEngine_70_30"),
            eq1.rename("Engine1_Carry"),
            eq2.rename("Engine2_Directional"),
            eq_spy.rename("SP500_SPY"),
            eq_qqq.rename("NASDAQ_QQQ"),
            eq_6040.rename("60_40_SPY_AGG"),
            eq_overlay.rename("BTC_Overlay_30pct"),
        ],
        axis=1
    ).dropna()

    rets = curves.pct_change().dropna()
    rf_d = daily_rate_from_apr(STABLE_APR)

    # Metrics
    metrics = []
    for col in curves.columns:
        eq = curves[col]
        r = rets[col]
        metrics.append(pd.Series({
            "CAGR": cagr(eq),
            "AnnVol": ann_vol(r),
            "Sharpe": sharpe(r, rf_daily=rf_d),
            "MaxDD": max_drawdown(eq),
            "Ulcer": ulcer_index(eq),
            "IR_vs_SPY": info_ratio(r, rets["SP500_SPY"]) if col != "SP500_SPY" else np.nan
        }, name=col))
    metrics_df = pd.DataFrame(metrics)

    # Save evidence
    curves.to_csv(os.path.join(OUT_DIR, "equity_curves_two_engines.csv"))
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_two_engines.csv"))

    plt.figure()
    (curves / curves.iloc[0]).plot()
    plt.title("Equity Curves (Normalized) — Two-Engine Portfolio (70/30)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "equity_curves_two_engines.png"), dpi=150)
    plt.close()

    print("\n=== METRICS (Two-Engine: 70% Carry + 30% Directional) ===")
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print(metrics_df)
    print(f"\nSaved evidence to: {OUT_DIR}/ (csv + png)\n")

    # Extra: sanity check realized Sharpe of engines
    print("Realized Sharpe (Engine1):", round(sharpe(rets["Engine1_Carry"], rf_daily=rf_d), 3))
    print("Realized Sharpe (Engine2):", round(sharpe(rets["Engine2_Directional"], rf_daily=rf_d), 3))
    print("Realized Sharpe (TwoEngine):", round(sharpe(rets["TwoEngine_70_30"], rf_daily=rf_d), 3))


if __name__ == "__main__":
    main()



