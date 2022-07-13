import random

class Die:
    def __init__(self, sides = 6):
        self.sides = sides
    def roll_die(self):
        print(random.randint(1,self.sides))

D6 = Die()
for i in range(10):
    D6.roll_die()

D20 = Die(20)
for i in range(10):
    D20.roll_die()
