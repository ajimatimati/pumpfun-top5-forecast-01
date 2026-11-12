# pumpfun-top5-forecast-01

Calibrated probabilistic forecaster for the top 5 Solana pump.fun coins by market cap across multi-hour horizons.

## Methodology
1. **Venue Differentiation**: Differentiates graduated PumpSwap pools with locked canonical liquidity from bonding curves.
2. **Hazard-Rate Survival Dynamics**: Calibrates an age and market-cap dependent crash hazard for incumbents.
3. **Stochastic Multi-Horizon Monte Carlo Simulation**: Simulates price diffusion, momentum persistence, and newcomer arrival processes across all answer read horizons.
4. **Bayes-Optimal Allocation**: Derives place marginals and unnamed newcomer probability masses directly matching the proper log scoring rule.
5. **Calibrated Size Distributions**: Smooths empirical order-statistic distributions over 82 logarithmic market-cap bins.
6. **Top 5 Agreement Maximizer**: Solves the maximum expected agreement assignment for the tie-breaker.
