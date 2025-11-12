import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://satellite-api.open-meteo.com/v1/archive"
params = {
	"latitude": [51.0267, NaN],
	"longitude": [7.5693, NaN],
	"hourly": "shortwave_radiation",
	"models": "satellite_radiation_seamless",
	"utm_source": "chatgpt.com",
}
responses = openmeteo.weather_api(url, params=params)

# Process 2 locations
for response in responses:
	print(f"\nCoordinates: {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Elevation: {response.Elevation()} m asl")
	print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
	
	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_shortwave_radiation = hourly.Variables(0).ValuesAsNumpy()
	
	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}
	
	hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
	
	hourly_dataframe = pd.DataFrame(data = hourly_data)
	print("\nHourly data\n", hourly_dataframe)
	