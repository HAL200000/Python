people=['a','b','c']
dic={
    'a':'c++',
    'b':'c',
    'c':'python',
    'd':'jvav',
    }
for name in dic.keys():
    if name in people:
        print(f"tks,{name}")
    else:
        print (f"plz join us,{name}")
    
