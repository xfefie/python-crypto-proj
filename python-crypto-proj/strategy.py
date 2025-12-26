# strategy.py
from __future__ import annotations

import numpy as np
import pandas as pd

class SleeveEngineStrategy:
    """
    Regime-gated BTC + gated alt sleeve + stable carry + vol targeting.
    Outputs target weights for: ["CASH", "BTC", "ALT"] each rebalance date.

    Notes:
    - "CASH" is a stablecoin/carry sleeve proxy.
    - BTC is typically a BTC ETF (e.g., IBIT/BITO) in backtests.
    - ALT is a quality alt proxy (e.g., ETH ETF or ETH-USD).
    """

    def __init__(
        self,
        vol_target_annual: float = 0.18,
        max_btc_w: float = 0.35,
        max_alt_w: float = 0.20,
        min_cash_w: float = 0.40,
        trend_fast: int = 50,
        trend_slow: int = 200,
        vol_lookback: int = 20,
        risk_off_vol_z: float = 1.5,
    ):
        self.vol_target_annual = vol_target_annual
        self.max_btc_w = max_btc_w
        self.max_alt_w = max_alt_w
        self.min_cash_w = min_cash_w
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self.vol_lookback = vol_lookback
        self.risk_off_vol_z = risk_off_vol_z

    @staticmethod
    def _realized_vol(daily_rets: pd.Series, lookback: int) -> pd.Series:
        # annualized vol estimate
        return daily_rets.rolling(lookback).std() * np.sqrt(252)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        prices columns expected: ["BTC", "ALT"]  (CASH handled separately)
        Returns weights DataFrame columns: ["CASH","BTC","ALT"]
        """
        px = prices.copy().dropna()
        btc = px["BTC"]
        alt = px["ALT"]

        # Trend regime: risk-on when BTC fast MA > slow MA
        ma_fast = btc.rolling(self.trend_fast).mean()
        ma_slow = btc.rolling(self.trend_slow).mean()
        trend_on = (ma_fast > ma_slow).astype(float)

        # Vol regime: risk-off when BTC vol spikes vs its own history
        btc_rets = btc.pct_change()
        btc_vol = self._realized_vol(btc_rets, self.vol_lookback)
        vol_z = (btc_vol - btc_vol.rolling(252).mean()) / (btc_vol.rolling(252).std())
        vol_ok = (vol_z < self.risk_off_vol_z).astype(float)

        # Primary gate: only take risk when trend_on and vol_ok
        risk_gate = (trend_on * vol_ok).clip(0, 1)

        # Base target (before vol targeting)
        # - In risk-on: allocate to BTC + small ALT sleeve
        # - In risk-off: allocate mostly CASH
        btc_base = self.max_btc_w * risk_gate
        alt_base = self.max_alt_w * risk_gate

        # Additional gating for ALT: only when BTC trend is on AND ALT momentum positive
        alt_mom = (alt / alt.shift(63) - 1.0)  # ~3 months
        alt_gate = ((alt_mom > 0).astype(float) * trend_on).clip(0, 1)
        alt_base = alt_base * alt_gate

        # Vol targeting: scale BTC+ALT so total "risk sleeve" meets target
        risk_sleeve = btc_base + alt_base
        # Use BTC vol as proxy for sizing (conservative)
        scale = (self.vol_target_annual / (btc_vol + 1e-12)).clip(0.0, 2.0)
        risk_sleeve_scaled = (risk_sleeve * scale).clip(0.0, 0.60)  # cap total risky exposure

        # Split scaled risk between BTC and ALT proportionally
        prop = (btc_base + alt_base).replace(0, np.nan)
        btc_w = (btc_base / prop) * risk_sleeve_scaled
        alt_w = (alt_base / prop) * risk_sleeve_scaled
        btc_w = btc_w.fillna(0.0)
        alt_w = alt_w.fillna(0.0)

        # Cash = remainder, but enforce min_cash_w
        cash_w = 1.0 - (btc_w + alt_w)
        # enforce min cash by shrinking risky weights proportionally if needed
        short_cash = (self.min_cash_w - cash_w).clip(lower=0.0)
        shrink = (1.0 - short_cash / (btc_w + alt_w + 1e-12)).clip(0.0, 1.0)
        btc_w *= shrink
        alt_w *= shrink
        cash_w = 1.0 - (btc_w + alt_w)

        w = pd.DataFrame({"CASH": cash_w, "BTC": btc_w, "ALT": alt_w}, index=px.index)
        return w
