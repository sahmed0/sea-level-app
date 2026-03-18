import pandas as pd
from prophet import Prophet
from config import MAX_PREDICTION_YEAR
import warnings
warnings.filterwarnings("ignore")

def main():
    """
    Pre-computes the Prophet forecast and saves it to a CSV file.
    This script is intended to be run in an environment where Prophet can be
    installed, such as a CI/CD pipeline, to generate the forecast data
    that the main web application will use.
    """
    print("Pre-computing Prophet forecast...")
    
    # Load data using the same function as the main application
    from data import loadData
    df = loadData()

    # Prepare dataframe for Prophet
    # It requires columns named 'ds' (datestamp) and 'y' (value).
    df_prophet = df.rename(columns={
        'Year': 'ds',
        'CSIRO Adjusted Sea Level': 'y'
    })
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y').astype('datetime64[s]')

    # Instantiate and fit the model. 
    # Seasonality is kept simple as we are dealing with yearly data.
    model = Prophet(
        yearly_seasonality=True,
        daily_seasonality=False,
        weekly_seasonality=False
    )
    model.fit(df_prophet)

    # Create a future dataframe for predictions up to the max year
    future = model.make_future_dataframe(
        periods=(MAX_PREDICTION_YEAR - df['Year'].max()),
        freq='YE'
    )
    future['ds'] = future['ds'].astype('datetime64[s]')

    # Make the forecast
    forecast = model.predict(future)

    # Save the relevant columns of the forecast to a CSV file
    forecast_to_save = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_to_save.to_csv("prophet_forecast.csv", index=False)
    
    print("Forecast successfully saved to prophet_forecast.csv")

if __name__ == "__main__":
    main()
