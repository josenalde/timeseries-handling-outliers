# timeseries-handling-outliers
<span style="color:red">*You are a machine learning expert with high expertise on statistics. You are creating a forecasting model for revenue sales prediction, with weekly, montly and quarter horizonts, trained with three year historical data. The series was made statitionary with difference technique. However this historical data has "good outliers" representing accumulative sales from companies, representing sudden "good" increase in total sales amount. You were requested to not remove these outliers, but somehow incorporate them into your model. How can you do that?*</span>

Great question — and it’s a very common, nuanced issue in forecasting. You have good outliers (structural, meaningful spikes that correspond to real business events like bulk/accumulated purchases) and you’ve been asked not to remove them. The challenge is:
- If you treat them as “noise,” your model will underfit these events and underpredict future surges.
- If you let them dominate the training signal without structure, your forecasts will be unstable.
Here’s a structured way to incorporate “good outliers” into your forecasting model:

## 1) When you known when these spikes may happen
1.1) Feature Engineering: Turn Outliers into Signals
Instead of treating spikes as anomalies, model them as informative features.
Create a binary or categorical flag indicating periods where accumulated sales occurred.
Example: accumulated_event = 1 for weeks/months with unusual surges.

If you can identify the reason (e.g., quarterly settlements, corporate orders, promotional campaigns), encode that explicitly as a regressor.
This lets the model learn the conditional pattern: “if event → expect spike.”

1.2) Use Models That Support Exogenous Variables
Choose forecasting models that can incorporate explanatory regressors:
- SARIMAX (ARIMA with exogenous variables) → works well with differenced stationary data and event indicators.
- Prophet → supports holiday/event regressors, great for recurring spikes.
- Machine learning regressors (XGBoost, LGBM, Random Forests) → allow event features and interactions with seasonality/time.
- Neural nets (N-BEATS, DeepAR, TFT) → can also use covariates and capture nonlinear impact of outliers.

1.3) Handle Scale: Decompose “Normal” vs “Event” Sales
Model the series as a combination of two processes:
Baseline_t = smooth component (trend, seasonality, autoregressive dynamics).
EventImpact_t = modeled separately (binary flag, distribution of outlier sizes, or even a regression on known event drivers).
This avoids contaminating your baseline dynamics with spikes.

1.4) Distributional Forecasting
Instead of predicting only the mean, use models that estimate uncertainty or the full distribution:
- Quantile regression (LightGBM, CatBoost with quantile loss).
- Bayesian models (hierarchical time series, state-space).
- These allow your forecasts to reflect the fact that “sometimes there are spikes.”

### SARIMAX with an “event flag” as an exogenous variable.
Imagine you have weekly revenue data over 3 years (sales), and you know which weeks had “accumulative events.”

a) SARIMAX with Event Flag
```
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Example data
np.random.seed(42)
weeks = pd.date_range("2021-01-01", periods=156, freq="W")  # 3 years weekly
sales = 200 + np.random.normal(0, 10, size=156).cumsum()  # baseline growth

# Add some "good outliers" (accumulated sales)
event_weeks = np.random.choice(range(156), size=10, replace=False)
sales[event_weeks] += np.random.randint(100, 300, size=10)

df = pd.DataFrame({"date": weeks, "sales": sales})
df["event_flag"] = 0
df.loc[event_weeks, "event_flag"] = 1

# Make series stationary (difference if needed)
y = df["sales"].diff().dropna()
X = df["event_flag"].iloc[1:]  # align with differenced series

# SARIMAX model
model = sm.tsa.SARIMAX(y, exog=X, order=(1,0,1), seasonal_order=(0,0,0,0))
results = model.fit(disp=False)

print(results.summary())

# Forecast next 12 weeks
future_events = np.zeros(12)  # assume no events
forecast = results.get_forecast(steps=12, exog=future_events)
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int()

print(pred_mean.head())
```
👉 Here, the event_flag allows the model to treat outliers not as noise but as structured shifts.

b) Gradient Boosting (XGBoost)
```
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Feature engineering
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter

# Lag features (autoregression)
for lag in [1, 2, 4, 12]:
    df[f"lag_{lag}"] = df["sales"].shift(lag)

df = df.dropna()

X = df[["weekofyear", "month", "quarter", "event_flag", "lag_1", "lag_2", "lag_4", "lag_12"]]
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=12)

# Train model
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

# Forecast last 12 weeks
y_pred = model.predict(X_test)

print("Last predictions:", y_pred)
```
👉 In this version, the event_flag + lag features let the ML model explicitly learn the spikes without distorting baseline seasonality.
✅ Practical approach I’d recommend:
Engineer an “event flag” when spikes occur (and, if possible, a forward-looking indicator).
Train a SARIMAX or Gradient Boosted Trees model with these flags as regressors.

## 2) If events are not predictable,:
use a probabilistic model that allows heavy-tailed error distributions (e.g., Student-t innovations in ARIMA, or quantile regression forests). If you don’t have explicit covariates to anticipate them:
- Consider mixture models (e.g., two regimes: normal sales vs. accumulated sales).
- Markov-switching state-space models or regime-switching ARIMA can help.
This way, the model learns “most of the time baseline, sometimes spike regime.”
This way, you respect the business request to keep the outliers, but you also prevent them from corrupting the baseline dynamics, while still allowing your forecasts to reflect the possibility of “good” spikes.
### If events are not predictable in advance, you could model their probability instead (mixture models, heavy-tailed error distributions, quantile regression).
👉 You don’t know the future event flags (company can’t tell you when “accumulative sales” will happen).
That means you cannot just feed a binary event_flag into the forecast horizon. Instead, you need to make your model robust to spikes and allow them to show up probabilistically.
Here are three strategies:

2.1) Use Heavy-Tailed Error Distributions
If the “good outliers” are rare but impactful, a normal error assumption will underestimate them.
Instead, use a distribution that tolerates fat tails:

- ARIMA with Student-t innovations (instead of Gaussian).
- Bayesian Structural Time Series (BSTS) with heavy-tailed priors.
- Quantile regression models (XGBoost/LightGBM with quantile loss).

➡️ This way, the model doesn’t need to know exactly when spikes happen, but it widens the forecast distribution to include them.

Example (statsmodels SARIMAX with Student-t):
```
# Fit SARIMAX with t-distributed errors
from arch.univariate import ARX, StudentsT

model = ARX(df["sales"].dropna(), lags=2, constant=True, rescale=True)
model.distribution = StudentsT()   # heavy tails
res = model.fit()

print(res.summary())
```
Here, the Student-t innovation helps capture outliers as part of the expected distribution rather than “breaking” the model.

2.2) Probabilistic Forecasting with Regime Switching
Model sales as two regimes:
- Normal regime (baseline demand).
- Spike regime (accumulated sales events).
- A Markov-Switching Model (or Hidden Markov Model) can learn the probability of being in each regime, even if you don’t know the future events explicitly.
Example (statsmodels Markov Switching):
```
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
# Fit a two-regime Markov switching model

mod = MarkovRegression(df["sales"], k_regimes=2, trend='c', switching_variance=True)
res = mod.fit()
print(res.summary())
# Get regime probabilities
probs = res.smoothed_marginal_probabilities
df["regime_prob_spike"] = probs[1]
# Here, regime 1 might represent “normal sales” and regime 2 the “spike weeks.”
# The forecast is then weighted by the probability of being in each regime.
```

2.3) Forecast Two Components Separately
Instead of lumping everything together:
- Baseline_t → modeled with ARIMA, Prophet, or ML (smooth part).
- Spikes_t → modeled as a counting process (frequency of events) × magnitude distribution.
Example: fit a Poisson/Negative Binomial for how often spikes happen.
Fit a distribution (e.g., Gamma or LogNormal) for how big they are.
Then simulate future sales = baseline + random spikes.
This is a stochastic simulation approach — your forecasts are scenario-based (with/without spikes), giving decision-makers a realistic uncertainty range.

2.4) Two regime Markov
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# ------------------------
# 1. Generate synthetic sales data
# ------------------------
np.random.seed(42)
weeks = pd.date_range("2021-01-01", periods=156, freq="W")

# baseline trend + noise
baseline = 200 + np.cumsum(np.random.normal(0, 5, 156))

# introduce "good outliers" (accumulative spikes)
sales = baseline.copy()
event_weeks = np.random.choice(range(156), size=12, replace=False)
sales[event_weeks] += np.random.randint(80, 200, size=12)

df = pd.DataFrame({"date": weeks, "sales": sales})

# ------------------------
# 2. Fit a 2-regime Markov Switching model
# ------------------------
mod = MarkovRegression(df["sales"], k_regimes=2, trend="c", switching_variance=True)
res = mod.fit()

print(res.summary())

# ------------------------
# 3. Get regime probabilities
# ------------------------
df["prob_regime0"] = res.smoothed_marginal_probabilities[0]  # normal regime
df["prob_regime1"] = res.smoothed_marginal_probabilities[1]  # spike regime

# ------------------------
# 4. Visualization
# ------------------------
fig, ax = plt.subplots(2, 1, figsize=(12,6), sharex=True)

# Sales series
ax[0].plot(df["date"], df["sales"], label="Sales", color="black")
ax[0].set_title("Sales with Accumulative Spikes (Good Outliers)")
ax[0].legend()

# Regime probabilities
ax[1].plot(df["date"], df["prob_regime0"], label="P(Normal Regime)", color="blue")
ax[1].plot(df["date"], df["prob_regime1"], label="P(Spike Regime)", color="red")
ax[1].fill_between(df["date"], 0, df["prob_regime1"], color="red", alpha=0.2)
ax[1].set_title("Smoothed Probabilities of Being in Spike Regime")
ax[1].legend()

plt.tight_layout()
plt.show()
```
2.5) Regime switching
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# ------------------------
# 1. Generate synthetic sales data
# ------------------------
np.random.seed(42)
weeks = pd.date_range("2021-01-01", periods=156, freq="W")

baseline = 200 + np.cumsum(np.random.normal(0, 5, 156))
sales = baseline.copy()
event_weeks = np.random.choice(range(156), size=12, replace=False)
sales[event_weeks] += np.random.randint(80, 200, size=12)

df = pd.DataFrame({"date": weeks, "sales": sales})

# ------------------------
# 2. Fit 2-regime Markov Switching model
# ------------------------
mod = MarkovRegression(df["sales"], k_regimes=2, trend="c", switching_variance=True)
res = mod.fit()

print(res.summary())

# ------------------------
# 3. Get smoothed probabilities
# ------------------------
df["prob_regime0"] = res.smoothed_marginal_probabilities[0]
df["prob_regime1"] = res.smoothed_marginal_probabilities[1]

# ------------------------
# 4. Forecast next 12 weeks
# ------------------------
n_forecast = 12
last_state_probs = res.smoothed_marginal_probabilities.iloc[-1].values  # [P(reg0), P(reg1)]

# Regime-specific means (constant terms)
means = res.params[[0, 1]]  # intercepts of each regime

trans_mat = mod.regime_transition_matrix(res.params)   # <-- método do modelo, não do resultado
# se time-varying: shape (k,k,nobs), senão (k,k)
if trans_mat.ndim == 3:
    # escolha a matriz no último instante observado para forecasting
    P = trans_mat[:, :, -1]
else:
    P = trans_mat

# Forecast storage
forecast_dates = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(weeks=1), periods=n_forecast, freq="W")
forecast_sales = []
forecast_probs = [last_state_probs]

probs = last_state_probs
for _ in range(n_forecast):
    # Expected sales = weighted avg of regime means
    forecast_sales.append(np.dot(probs, means))
    # Update regime probabilities for next step
    probs = probs @ P
    forecast_probs.append(probs)

forecast_df = pd.DataFrame({
    "date": forecast_dates,
    "forecast_sales": forecast_sales,
    "prob_regime0": [p[0] for p in forecast_probs[1:]],
    "prob_regime1": [p[1] for p in forecast_probs[1:]],
})

# ------------------------
# 5. Visualization
# ------------------------
fig, ax = plt.subplots(2, 1, figsize=(12,7), sharex=True)

# Historical + forecast sales
ax[0].plot(df["date"], df["sales"], label="Historical Sales", color="black")
ax[0].plot(forecast_df["date"], forecast_df["forecast_sales"], label="Forecast", color="green", marker="o")
ax[0].set_title("Markov Switching Forecast (Normal vs Spike Regimes)")
ax[0].legend()

# Regime probabilities
ax[1].plot(df["date"], df["prob_regime1"], label="P(Spike Regime, Historical)", color="red")
ax[1].plot(forecast_df["date"], forecast_df["prob_regime1"], label="P(Spike Regime, Forecast)", color="orange", linestyle="--")
ax[1].fill_between(df["date"], 0, df["prob_regime1"], color="red", alpha=0.2)
ax[1].fill_between(forecast_df["date"], 0, forecast_df["prob_regime1"], color="orange", alpha=0.2)
ax[1].set_title("Probability of Spike Regime")
ax[1].legend()

plt.tight_layout()
plt.show()
```
