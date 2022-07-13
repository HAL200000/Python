result = {}
flag = True

while flag:
    name = input("what's your name?(print 'quit' to quit the program)\n")
    if name == 'quit': break
    place = input("where would you go?(print 'quit' to quit the program)\n")
    if place == 'quit': break

    result[name] = place

for name, place in result.items():
    print(f"{name} wants to visit {place}\n")