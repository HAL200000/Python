sandwich_orders = ['vegetable','pastrami','tuna','pastrami','meat','pastrami','elephant']
finished_sandwitches = []
print('All of the pastrami sandwiches has been sold.')
while 'pastrami' in sandwich_orders :
    sandwich_orders.remove('pastrami')

while sandwich_orders :
    currrent_sandwich = sandwich_orders.pop()
    print(f'I made your {currrent_sandwich} sandwitch.')
    finished_sandwitches.append(currrent_sandwich)

print(finished_sandwitches)