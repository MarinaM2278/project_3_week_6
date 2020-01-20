import requests

url = 'http://localhost:5000/predict_api'

r = request.post(url, json = {
        'name': ["BMW"],
        'location': ['location_Bangalorerd'],
        'year':  [2000],
        'kilometers_driven' : [120000],
        'fuel_type': [Diesel],
        'transmission' : [1],
        'owner_type' : [2],
        'seats': [5]
})
print(r.json())
