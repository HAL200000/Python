cities = {
    'A': {
        'a1': 1,
        'a2': 2,
    },
    'B': {
        'b1': '1b',
        'b2': '2b',
    },
    'C': {
        'c1': '1c',
        'c2': '2c',
    }
}

for city_name, city_info in cities.items():
    print(f"The city's name is {city_name}\n")
    for k,v in city_info.items():
        print(f"{k}:  {v}\n")
