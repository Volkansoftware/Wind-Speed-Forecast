import io
import base64
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, send_file, make_response
import matplotlib.dates as mdates  

def create_wind_speed_forecast_plot():
    city = "Istanbul"
    api_key = "74cf99eac1864eaba4b76c213d93c173"
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&cnt=40&appid={api_key}"
    response = requests.get(url)
    data = json.loads(response.text)

    weather_data = []
    for item in data["list"]:
        date_str = item["dt_txt"]
        date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        temp = item["main"]["temp"] - 273.15
        pressure = item["main"]["pressure"]
        humidity = item["main"]["humidity"]
        wind_speed = item["wind"]["speed"]
        wind_direction = item["wind"]["deg"]
        weather_data.append({
            "date": date,
            "temp": temp,
            "pressure": pressure,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction
        })

    df = pd.DataFrame(weather_data)
    df["hour"] = df["date"].apply(lambda x: x.hour)
    df["day"] = df["date"].apply(lambda x: x.day)
    df["month"] = df["date"].apply(lambda x: x.month)
    df["wind_direction_cos"] = df["wind_direction"].apply(lambda x: round(math.cos(math.radians(x)), 2))
    df["wind_direction_sin"] = df["wind_direction"].apply(lambda x: round(math.sin(math.radians(x)), 2))

    X = df[["hour", "day", "month", "temp", "pressure", "humidity", "wind_direction_cos", "wind_direction_sin"]]
    y = df["wind_speed"]
    dates = df["date"]

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state=40)

    reg = LinearRegression().fit(X_train, y_train)

    scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

    print(f'Cross-validated scores: {scores}')
    print(f'Average MAE: {scores.mean()}')

    y_pred = reg.predict(X_test)

    diff = abs(y_test - y_pred)
    error_rate = sum(diff) / sum(y_test)
    print(f"Test set error rate: {error_rate:.2f}")

    # Create a new DataFrame to hold the date, actual and predicted values
    result_df = pd.DataFrame({"date": dates_test, "actual": y_test, "predicted": y_pred})

    # Sort the DataFrame by date
    result_df = result_df.sort_values(by="date")
    five_days_later = datetime.now() + timedelta(days=5)
    result_df = result_df[result_df["date"] <= five_days_later]

    plt.plot(result_df["date"], result_df["predicted"], label="predicted")
    plt.plot(result_df["date"], result_df["actual"], label="openweathermap")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))  # <-- add this line
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # <-- add this line
    plt.xlabel("Date")
    plt.ylabel("Wind Speed (m/s)")
    plt.title(f"Wind Speed Forecast for {city}")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', city="Istanbul")

@app.route('/plot')
def plot():
    buf = create_wind_speed_forecast_plot()
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)