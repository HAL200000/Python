def make_car(maker, type, **car):
    car['maker'] = maker
    car['type'] = type

    return car

print(make_car('subaru','outback', color='blue', tow_package=True))