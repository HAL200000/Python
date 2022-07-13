my_pizza=['a','b','c','d','e']
fr_pizza=my_pizza[:]
my_pizza.append('f')
fr_pizza.append('g')
print("my favorite pizzas are:",end=' ')
for pizza in my_pizza:
    print(pizza,end=' ')
print('\n')
print("my friends' favorite pizzas are:",end=' ')
for pizza in fr_pizza:
    print(pizza,end=' ')
