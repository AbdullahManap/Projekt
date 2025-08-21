import requests

API_KEY = "5605fc8cf0d2b91d22a620881533b0c4"

lat = "51.0205" # breitengrad Bergneustadt
lon = "7.6486" # längengrad Bergneustadt

requests_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
response = requests.get(requests_url)

if response.status_code == 200:
    data = response.json()
    #print(data)
    weather_description1 = data["list"][0]["weather"][0]["description"]
    print(weather_description1)
    temperatur1 = round(data["list"][0]["main"]["temp"] - 273.15)
    print(temperatur1)
    weather_description2 = data["list"][0]["weather"][0]["description"]
    print(weather_description2)
    temperatur2 = round(data["list"][1]["main"]["temp"] - 273.15)
    print(temperatur2)
    weather_description3 = data["list"][0]["weather"][0]["description"]
    print(weather_description3)
    temperatur3 = round(data["list"][2]["main"]["temp"] - 273.15)
    print(temperatur3)
    weather_description4 = data["list"][0]["weather"][0]["description"]
    print(weather_description4)
    temperatur4 = round(data["list"][3]["main"]["temp"] - 273.15)
    print(temperatur4)
