import pandas as pd
from config import MAX_PREDICTION_YEAR
from models import getProphetForecast
import warnings
warnings.filterwarnings("ignore")

def main():
    print("Pre-computing Prophet forecast...")
    from data import loadData
    df = loadData()
    
    # Run the forecast using the local Prophet installation
    forecast = getProphetForecast(df, MAX_PREDICTION_YEAR)
    
    # Save the output to a CSV file
    forecast.to_csv("prophet_forecast.csv", index=False)
    print("Forecast successfully saved to prophet_forecast.csv")

if __name__ == "__main__":
    main()
