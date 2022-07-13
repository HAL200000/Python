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