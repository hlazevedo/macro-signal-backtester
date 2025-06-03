# Macro Signal Backtesting Framework

A comprehensive, modular Python framework for backtesting systematic macro trading strategies using economic indicators and market signals.

---

## Project Overview

This framework implements a systematic trading strategy based on yield curve dynamics. Built over a weekend with the help of coding assistants.

---

## Key Features

-   **Modular Architecture:** Clean OOP design with extensible base classes.
-   **Multiple Data Sources:** FRED economic data and Yahoo Finance price feeds.
-   **Signal Framework:** Extensible signal generation with normalization and filtering.
-   **Portfolio Management:** Complete NAV tracking, transaction costs, and performance metrics.
-   **Risk Management:** Drawdown monitoring and position sizing controls.
-   **Visualization:** Illustrative performance analysis and reporting.

---

## Strategy: Simple Yield Curve Signal

### Core Logic

-   Uses 10Y-2Y Treasury spread as primary macro signal.
-   Normalizes signal using rolling z-score (252-day window).
-   Implements threshold-based allocation:
    -   Signal > 0.5: 100% SPY (risk-on)
    -   Signal < -0.5: 100% TLT (risk-off)
    -   Neutral: 50/50 allocation

### Economic Rationale

Yield curve slope reflects market expectations for growth and inflation. Steepening curves typically signal economic expansion (favoring equities), while flattening/inversion often precedes recessions (favoring bonds).

---

## Performance Summary (2015-2024)

- **Total Return**: 40% over 9 years ($1M → $1.4M)
- **Maximum Drawdown**: 25.88% (Q4 2022 - Q2 2024)
- **Key Success**: Excellent timing during Q3 2020 - Q4 2022 equity rally (100% SPY allocation)
- **Timing Challenges**: 
  - Missed 2016-2019 equity bull market (mostly TLT allocation)
  - Recent drawdown from Q4 2022 rotation to bonds during continued rate hike environment
- **Signal Behavior**: Yield curve signal shows lag at major turning points, highlighting need for multi-signal approaches

**Allocation Timeline Analysis:**
- 2015: 50/50 baseline allocation
- 2016-2019: Predominantly bonds (missed equity gains)
- 2020-2022: 100% equities (captured post-COVID rally - major performance driver)
- 2023-2024: Back to bonds (coincided with drawdown period)

---

## Architecture

```text
macro-signal-backtester/
├── src/
│   ├── data/           # Data loading and management
│   ├── signals/        # Signal generation classes
│   ├── strategy/       # Trading strategy implementation
│   ├── portfolio/      # Portfolio tracking and metrics
│   ├── backtest/       # Backtesting engine and analysis
│   └── utils/          # Logging and validation utilities
├── examples/           # Usage examples and strategy demos
├── tests/              # Unit tests
├── config/             # Configuration files
└── notebooks/          # Jupyter analysis notebooks
```
## Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/[hlazevedo]/macro-signal-backtester.git
cd macro-signal-backtester
```

2. **Create virtual environment**
```bash 
python -m venv venv
source venv/bin/activate 
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4.  **Configure API access:**
```bash
# Copy the example config file
cp config/config.yaml.example config/config.yaml
```
Edit `config/config.yaml` with your FRED API key.
(Get a free API key at: [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html))

5.  **Run yield curve strategy example:**
```bash
python examples/simple_yield_curve_strategy.py
```