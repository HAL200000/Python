class Restaurant:
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type
        self.number_served = 0

    def describe_restaurant(self):
        print(f"the restaurant's name is {self.restaurant_name}, and its cuisine type is {self.cuisine_type}")

    def open_restaurant(self):
        print("the restaurant is now opening!")

    def set_number_served(self, num):
        self.number_served = num

    def increment_number_served(self, add_num):
        self.number_served += add_num

class IceCreamStand(Restaurant):
    def __init__(self, restaurant_name, cuisine_type):
        super(IceCreamStand, self).__init__(restaurant_name, cuisine_type)
        self.flavors = []

    def show_flavors(self):
        print(self.flavors)

restaurant = Restaurant('KFC', 'fried_chicken')

print(restaurant.restaurant_name)
print(restaurant.cuisine_type)

restaurant.describe_restaurant()
restaurant.open_restaurant()

restaurant.set_number_served(233)
print(restaurant.number_served)

restaurant.increment_number_served(114514)
print(restaurant.number_served)

new_ice_cream_stand = IceCreamStand('MXBC', 'icecream')
new_ice_cream_stand.flavors = ['chocolate', 'strawberry', 'origin']
new_ice_cream_stand.describe_restaurant()
new_ice_cream_stand.show_flavors()