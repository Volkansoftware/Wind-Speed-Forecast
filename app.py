import io
import base64
from flask import Flask, render_template, send_file
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error





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

# Yeni özellikler oluşturma
    df["hour"] = df["date"].apply(lambda x: x.hour)
    df["day"] = df["date"].apply(lambda x: x.day)
    df["month"] = df["date"].apply(lambda x: x.month)
    df["wind_direction_cos"] = df["wind_direction"].apply(lambda x: round(math.cos(math.radians(x)), 2))
    df["wind_direction_sin"] = df["wind_direction"].apply(lambda x: round(math.sin(math.radians(x)), 2))

# Mevcut verilere dayalı olarak rüzgar hızını tahmin etme
    X = df[["hour", "day", "month", "temp", "pressure", "humidity", "wind_direction_cos", "wind_direction_sin"]]
    y = df["wind_speed"]
    reg = LinearRegression().fit(X, y)

# Son 5 günün verisini oluşturma
    last_five_days = [datetime.now() - timedelta(days=i) for i in range(4, -1, -1)]
    last_five_days_df = pd.DataFrame()
    for day in last_five_days:
        for hour in range(0, 24, 3):
            date = datetime(day.year, day.month, day.day, hour)
            last_five_days_df = last_five_days_df.append({
                "date": date,
                "hour": hour,
                "day": day.day,
                "month": day.month
        }, ignore_index=True)

# Gerçek verileri API'den alma
    api_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}"
    api_response = requests.get(api_url)
    api_data = json.loads(api_response.text)

    api_weather_data = []
    for item in api_data["list"]:
        date_str = item["dt_txt"]
        date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        temp = item["main"]["temp"] - 273.15
        pressure = item["main"]["pressure"]
        humidity = item["main"]["humidity"]
        wind_speed = item["wind"]["speed"]
        wind_direction = item["wind"]["deg"]
        api_weather_data.append({
            "date": date,
            "temp": temp,
            "pressure": pressure,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction
    })

    api_df = pd.DataFrame(api_weather_data)


    api_df["hour"] = api_df["date"].apply(lambda x: x.hour)
    api_df["day"] = api_df["date"].apply(lambda x: x.day)
    api_df["month"] = api_df["date"].apply(lambda x: x.month)
    api_df["wind_direction_cos"] = api_df["wind_direction"].apply(lambda x: round(math.cos(math.radians(x)), 2))
    api_df["wind_direction_sin"] = api_df["wind_direction"].apply(lambda x: round(math.sin(math.radians(x)), 2))

    X_test = api_df[["hour", "day", "month", "temp", "pressure", "humidity", "wind_direction_cos", "wind_direction_sin"]]
    y_pred = reg.predict(X_test)
    diff = abs(api_df["wind_speed"] - y_pred)
    error_rate = sum(diff) / sum(api_df["wind_speed"])
    print(f"Error rate: {error_rate:.2f}")


    plt.plot(api_df["date"], y_pred, label="predicted")
    plt.plot(api_df["date"], api_df["wind_speed"], label="actual")
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