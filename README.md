# HJM Pricer

A Python library for simulating interest rate dynamics and pricing fixed-income derivatives 
(caplets, swaptions, etc.) using the **Heath–Jarrow–Morton (HJM)** framework with 
**Musiela parameterization** and **Monte Carlo simulation**.

---

## Features

- **Forward Curve Construction**  
  - Handles yield data and tenor structure  
  - Rolling-window capability for local volatility estimation  
  - Supports bootstrapping from market data

- **Volatility Surface Building**  
  - `VolatilitySurface` fits factor models (e.g., PCA) to forward rate changes  
  - Rolling window covariance estimation for local volatility  
  - Produces fitted local volatilities for each date and tenor

- **HJM Drift Calculation**  
  - `get_HJM_drifts()` computes tenor-sensitive drifts from volatility surface  
  - Supports Musiela parameterization for forward curve aging term  
  - Integrates over volatilities using trapezoidal rule

- **Monte Carlo Simulation**  
  - Multi-path forward rate simulation using HJM dynamics  
  - Fully vectorized random noise generation per path and factor  
  - Supports arbitrary number of factors from volatility decomposition

---

## Project Structure
```
HJM_pricer/
│
├── data                     
    ├── CurveBuilder.py        # Constructs forward curves from yield data 
    ├── TsyYieldLoader.py      # Fetches and preprocesses Treasury yield curve data 
    └── fred_tenor_map.json
├── MCSimulation
    ├── Volatility.py          # calculates principal components in historical volatility
    ├── volSurface.py          # VolatilitySurface class (factor fitting, local vols)
    ├── drift.py               # HJM drift calculation with optional Musiela term
    ├── MonteCarlo.py          # MCSimulation class for forward rate paths
├── pricers
    ├── capfloor_black.py
├── instruments
    ├── capfloors.py    
├── demo.ipynb                 # demo of how to use the framework
└── README.md
```
Next Steps
- Caplet payoff calculation from simulated paths
- Calibration to real-world market data
- Support for swaptions and other HJM-compatible instruments
- Benchmark performance against analytical approximations

References
- Martinsky, O. (n.d.). HJM: Heath–Jarrow–Morton model. GitHub repository.
  Available at: https://github.com/omartinsky/HJM
- Heath, Jarrow, Morton (1992). Bond Pricing and the Term Structure of Interest Rates.
- Musiela, M. (1993). Stochastic PDEs and Term Structure Models.
- Brigo, D., & Mercurio, F. (2007). Interest Rate Models – Theory and Practice.
