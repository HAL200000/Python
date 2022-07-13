import random
num=random.randint(1,1000)
guess=int(input("you think the number is:"))
if guess>num:
    print("too big,the ans is %d" %num)
elif guess==num:
    print("Congratulations!")
else:
    print("too small,the ans is %d" %num)
