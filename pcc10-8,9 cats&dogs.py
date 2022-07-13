try:
    with open('cats.txt') as cats, open('dogs.txt') as dogs:
        cat_info = cats.readlines()
        dog_info = dogs.readlines()

except FileNotFoundError:
    print('file not found!')

else:
    for cats in cat_info:
        print(cats.strip())
    for dogs in dog_info:
        print(dogs.strip())