import requests
import json

location = {
"year": 2018.000000,
"day": 8.000000,
"minute": 30.000000,
"temp": 1.700000,
"rhum": 75.000000,
"prcp": 1.000000,
"wspd": 35.300000,
"coco": 15.000000,
"holiday_yes": 0.000000,
"sin_hour": 0.707107,
"cos_hour": 0.707107,
"sin_month": 0.866025,
"cos_month": 0.500000
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=location)
result = response.json()

print(json.dumps(result, indent=2))