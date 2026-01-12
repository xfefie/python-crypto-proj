# Crypto Quant Project (Portfolio Sleeves + Backtesting)

A research + backtesting framework for building **multi-sleeve crypto portfolios** (spot + derivatives/overlays), measuring performance (CAGR/Sharpe/IR/max drawdown/Ulcer), and iterating toward target risk/return profiles using **engines** (e.g., high-Sharpe non-directional carry + directional beta).

> **Disclaimer:** This repo is for research/education only. It is **not** financial advice. Crypto is highly risky. Backtests can be misleading (survivorship bias, look-ahead bias, liquidity/fees, funding, slippage, borrow constraints, liquidation risk).

---

## What’s Inside

### Core ideas
- **Sleeves**: independent strategy modules (carry, trend, basis, mean-reversion, defensive, etc.)
- **Engines**: weighted blends of sleeves to achieve a portfolio objective  
  - Example: Engine 1 (≈70% weight) = high-Sharpe / low-dd “non-directional carry”
  - Example: Engine 2 (≈30% weight) = directional “beta engine” for upside capture
- **Overlays** (optional): leverage targeting, volatility targeting, risk-parity, drawdown controls, regime filters

### Metrics
- CAGR, Volatility, Sharpe, Sortino
- Max Drawdown, Calmar
- **Ulcer Index**, Ulcer Performance Index (UPI)
- Information Ratio, tracking error (if benchmarking)
- Exposure diagnostics (gross/net, beta, turnover)

### Outputs
- Backtest summary tables
- Equity curve + drawdown curve
- Sleeve attribution & weights over time
