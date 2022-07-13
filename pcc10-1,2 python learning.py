with open("learning_python.txt") as lp:
    contents = lp.read()
    # read()之后把光标定位在文件末尾，因此下一个read()/readlines()将不能有效读取数据
    # 一个方法是使用filename.seek(0)重新定位光标到文件头部
    lp.seek(0)
    lines = lp.readlines()
    print(contents+'\n')
    lp.seek(0)
    for line in lp:
       print(line)
    print('\n')

txt = ""
for line in lines:
    # lstrip():截掉字符串左边的空格或指定字符。     rstrip():截掉字符串右边的空格或指定字符。      strip():截掉字符串两边的空格或指定字符。
    print(line.strip())
    txt += line.lstrip()
print('\n')

txt = txt.replace('python', 'C++')
print(txt)

