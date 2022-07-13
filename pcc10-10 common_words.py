with open('pg68466.txt', encoding='UTF-8') as book:
    lines = book.readlines()
    contents = ""
    for line in lines:
        contents += line.strip()
    print(contents.lower().count('the '))
