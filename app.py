import io

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math


import matplotlib.pyplot as plt
from flask import Flask, render_template, send_file
import matplotlib.dates as mdates  

def create_wind_speed_forecast_plot():
    city = "Istanbul"
    api_key = "74cf99eac1864eaba4b76c213d93c173"
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&cnt=40&appid={api_key}"
    response = requests.get(url)
    data = json.loads(response.text)

    weather_data = []
    for item in data["list"]:
        date_str = item["dt_txt"] #her bir ölçüm için tarih ve saat bilgisini bir string (metin) olarak alır.
        date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S") # aldığı tarih ve saat bilgisini Python'un anlayabileceği bir datetime nesnesine dönüştürür.
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

    df = pd.DataFrame(weather_data) # weather datayı dataframe dönüştürür
    print(df.head())
    df["hour"] = df["date"].apply(lambda x: x.hour)
    df["day"] = df["date"].apply(lambda x: x.day)
    df["month"] = df["date"].apply(lambda x: x.month)
    df["wind_direction_cos"] = df["wind_direction"].apply(lambda x: round(math.cos(math.radians(x)), 2))
    df["wind_direction_sin"] = df["wind_direction"].apply(lambda x: round(math.sin(math.radians(x)), 2))
    print(df.head())
    X = df[["hour", "day", "month", "temp", "pressure", "humidity", "wind_direction_cos", "wind_direction_sin"]]
    y = df["wind_speed"] # hedef değişken
    dates = df["date"] # tarihleri izlemek için oluşturuldu
    
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state=25)
# 25 değeri her seferinde aynı bölünmüş veriyi elde etmek için kullanılır, böylece sonuçlar yeniden üretilebilir.
    reg = LinearRegression().fit(X_train, y_train)

   

    y_pred = reg.predict(X_test) # daha önce eğitilmiş olan reg modelini kullanarak X_test veri setindeki rüzgar hızını tahmin eder.

    diff = abs(y_test - y_pred)
    error_rate = sum(diff) / sum(y_test)
    print(f"Test set error rate: {error_rate:.2f}")

    # test setindeki tarihler, gerçek rüzgar hızı değerleri ve tahmin edilen rüzgar hızı değerlerini içeren yeni bir DataFrame oluşturur.
    result_df = pd.DataFrame({"date": dates_test, "actual": y_test, "predicted": y_pred}) 
    print(result_df.head())
    # Sort the DataFrame by date
    result_df = result_df.sort_values(by="date") # date sütununa göre sıralanır 
    five_days_later = datetime.now() + timedelta(days=5) # 5 günün sonrasını verir 
    result_df = result_df[result_df["date"] <= five_days_later] # sadece 5 gün gözükecek şekilde filtreler 
    
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