favorite_places={
    'A':['a','aa','aaa'],
    'B':['bb','bbb'],
    'C':['ccc'],
}
for name,places in favorite_places.items():
    print(f'{name}\'s favorite places are:')
    for place in places:
        print(f'{place}')