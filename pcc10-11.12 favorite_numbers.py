import json

file_name = 'user_favorite_nums_info.json'

def store_nums():
    num = input('plz input your favorite number: ')
    with open(file_name, 'w') as f:
        json.dump(num, f)
        print('your favorite num has been stored!')
def read_nums():
    with open(file_name, 'r') as f:
        num = json.load(f)
        print(f"I remember your favorite num is {num}!")

try:
    read_nums()
except FileNotFoundError:
    store_nums()