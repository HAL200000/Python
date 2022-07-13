question="plz input your age\n(input 'quit' to end the program): "
age=""
while 1:
    age=input(question)
    if age!="quit":
        if int(age)<3:
            print("you are free")
        elif int(age)>12:
            print("your price is $15")
        else:
            print("your price is $12")
    else:
        break