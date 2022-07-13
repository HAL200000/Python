from random import choice

pool = [1,4,5,'a','e']
prize = [1,5,'a','e']

tries = []
tot = 0

def win():
    if tries == prize:
        print("you wins!")
        return True
    return False

while not win():
    tot += 1
    tries.clear()
    for i in range (len(prize)):
        tries.append(choice(pool))

print(tot)