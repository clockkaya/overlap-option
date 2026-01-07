# Interest Rate Derivatives Pricing Engine (HJM Framework)

This repository contains a quantitative finance implementation for pricing interest rate derivatives using a **Two-Factor Heath-Jarrow-Morton (HJM) Model**. The project focuses on calibrating forward rate dynamics to market data and pricing exotic options using Monte Carlo simulation.

## 🚀 Key Features

* **Two-Factor HJM Calibration:** Calibrates the volatility parameters ($a(t)$ and $\phi$) to market Caplet prices and correlation constraints using non-linear optimization (`scipy.optimize.minimize`).
* **Monte Carlo Simulation:** Generates 10,000+ sample paths for the forward rate term structure.
* **Exotic Option Pricing:** Supports pricing for complex derivatives, including:
    * Standard Caplets (Black's Model vs. HJM).
    * Digital Put Spread Options.
    * Floorlets.
    * Custom Spread Options based on the spread between 3-month and 1-year rates.
* **Greeks Calculation:** Numerical estimation of **Delta** (sensitivity to underlying rate and spread) and **Vega** (sensitivity to volatility parameters).
* **Hedging Analysis:** Evaluation of hedging costs using digital options and floorlet combinations.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:** * `NumPy` & `SciPy`: High-performance numerical computation and optimization.
    * `Pandas`: Data management and CSV exporting.
    * `Matplotlib` (Optional): For visualizing forward curve dynamics.

## 📊 Methodology

### 1. Forward Rate Dynamics
The model simulates the evolution of the forward rate $f(t, T)$ according to:
$$df(t, T) = \alpha(t, T)dt + \sigma_1(t, T)dW_1(t) + \sigma_2(t, T)dW_2(t)$$
where the drift $\alpha(t, T)$ is determined by the HJM no-arbitrage condition.

### 2. Calibration
The volatility structure is calibrated to minimize the Mean Squared Error (MSE) between simulated Caplet prices and market prices derived from Black’s Implied Volatility, subject to correlation constraints between short-term and long-term rates.

## 📂 Project Structure

* `pricing.py`: The core engine containing calibration, simulation, and pricing logic.
* `67184800.csv`: Generated output containing calibrated parameters, simulated forward curves, and model performance metrics (MSE/Correlation).

## 📈 Usage

To run the pricing engine and calibrate the model:

```bash
python pricing.py
```