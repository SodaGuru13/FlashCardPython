import requests
def current_weather():
    url = "http://api.openweathermap.org/data/2.5/weather?q=Orlando&units=imperial&APPID=60c61be3700d4eb3171fa87fc11670c3"
    r = requests.get(url)

    weather_json = r.json()
    print(weather_json)

    min = weather_json['main']['temp_min']
    max = weather_json['main']['temp_max']

    print("The circus' forecast is", min, "as the low and", max, "as the high")