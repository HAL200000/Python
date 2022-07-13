list=["A","B","C","D"]
for i in range (len(list)):
    print(f"{list[i]},plz join us for dinner")
    
print(f"btw,{list[0]} can't show up,so E takes his place")

list[0]="E"
for i in range (len(list)):
    print(f"{list[i]},plz join us for dinner")

print("YEAH,a larger table is found.")
list.insert(0,"F")
list.insert(2,"G")
list.append("H")

for i in range (len(list)):
    print(f"{list[i]},plz join us for dinner")

print("sorry,too many people")
while len(list)>2:
    del_name=list.pop()
    print(f"I'm regret to tell you that you can't come,{del_name}")
for i in range (len(list)):
    print(f"{list[i]},plz join us for dinner")

del list[1]
del list[0]
print(list)
