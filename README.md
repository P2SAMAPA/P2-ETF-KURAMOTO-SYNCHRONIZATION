# P2-ETF-KURAMOTO-SYNCHRONIZATION

## Kuramoto Model for ETF Synchronization Dynamics

### Concept

This repository applies the **Kuramoto model** from statistical physics to ETF selection.

$$
\frac{d\theta_i}{dt} = \omega_i + K\sum_j A_{ij}\sin(\theta_j - \theta_i)
$$

ETFs are treated as coupled oscillators with:
- **Phase** (\(\theta_i\)): Market position
- **Frequency** (\(\omega_i\)): Return momentum
- **Coupling** (\(A_{ij}\)): Correlation strength

### Synchronization Order Parameter

$$
R(t)e^{i\psi(t)} = \frac{1}{N}\sum_j e^{i\theta_j(t)}
$$

- \(R(t)\): Measures market synchronization
- **R high (sync)**: Systemic risk, crowded positioning → avoid
- **R low (dispersion)**: Relative-value opportunities → trade

### Trading Logic

| R(t) | Interpretation | Action |
|------|----------------|--------|
| > 0.7 | Synchronized | Cautious, avoid crowded positions |
| 0.3 - 0.7 | Transitioning | Watch for direction |
| < 0.3 | Dispersed | Favorable for ETF selection |

### Multi-Window Backtesting

Tests multiple windows (126, 252, 504, 756, 1008 days) to find optimal training period for each universe.

### Installation

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-KURAMOTO-SYNCHRONIZATION
cd P2-ETF-KURAMOTO-SYNCHRONIZATION
pip install -r requirements.txt# P2-ETF-KURAMOTO-SYNCHRONIZATION
