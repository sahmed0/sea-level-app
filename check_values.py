import pandas as pd
from models import getLinearPredictions, getPolyPredictions, getProphetForecast
from data import loadData

df = loadData()
targetYear = 2050

print(f"Results for {targetYear}:")

# Linear
_, _, linTargetVal, linUnc = getLinearPredictions(df, targetYear)
print(f"Linear: {linTargetVal:.4f} +/- {linUnc:.4f}")

# Poly
_, _, polyTargetVal, polyUnc = getPolyPredictions(df, targetYear)
print(f"Poly: {polyTargetVal:.4f} +/- {polyUnc:.4f}")

# Prophet
forecast = getProphetForecast(df, 2050)
targetDate = pd.Timestamp(year=targetYear, month=1, day=1)
prophetRow = forecast[forecast['ds'] == targetDate]
if not prophetRow.empty:
    p_val = prophetRow['yhat'].values[0]
    p_unc = (prophetRow['yhat_upper'].values[0] - prophetRow['yhat_lower'].values[0]) / 2
    print(f"Prophet: {p_val:.4f} +/- {p_unc:.4f}")
else:
    print("Prophet row not found")
