# 🌊 Sea Level Predictor (Streamlit App)

An interactive web application that forecasts global sea level rise using **Facebook Prophet**. Unlike simple linear regression, this app models non-linear trends and provides a confidence interval ("cone of uncertainty") for future predictions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sea-level-app-faqqps9amzyqpcsoxntfb4.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Library](https://img.shields.io/badge/Model-Facebook%20Prophet-orange)

## 🚀 Live Demo

**[Click here to launch the App](https://sea-level-app-faqqps9amzyqpcsoxntfb4.streamlit.app/)**

![App Screenshot](sea level app screenshot.png)

## 🧐 What does this app do?

This application takes historical sea level data (1880–Present) and allows users to:
1.  **Visualise History:** See the raw EPA/CSIRO data points.
2.  **Forecast the Future:** Use a time-series model (Prophet) to predict sea levels up to the year 2100.
3.  **Interact:** Use a slider to dynamically change the prediction window and see how the uncertainty grows over time.

## 📊 Comparison: Why Prophet?

| Feature | Basic Linear Regression | This App (Prophet) |
| :--- | :--- | :--- |
| **Trend Shape** | Straight Line Only | Curved / Accelerating Trends |
| **Uncertainty** | None | Calculates Confidence Intervals |
| **Seasonality** | Ignored | Can handle yearly cycles |
| **Accuracy** | Low (Underfits recent acceleration) | High (Adapts to changing rates) |

## 🛠️ Tech Stack

* **Streamlit**: For the interactive web interface.
* **Prophet**: For time-series forecasting.
* **Pandas**: For data manipulation.
* **Matplotlib**: For plotting the results.

## 💻 How to Run Locally

If you want to run this code on your own machine instead of the web:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/sea-level-predictor.git](https://github.com/your-username/sea-level-predictor.git)
    cd sea-level-predictor
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

* `app.py`: The main application code containing the UI and Prophet logic.
* `requirements.txt`: List of Python libraries required for Streamlit Cloud.
* `epa-sea-level.csv`: The historical dataset.

## 📉 Data Source
Global Average Absolute Sea Level Change, 1880-2014 from the US Environmental Protection Agency using data from CSIRO, 2015; NOAA, 2015.
