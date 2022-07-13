with open('guest.txt', 'w') as guest:
    while True:
        name = input('plz enter your name:(print "q" to quit)\n')
        if name == 'q' :
            break
        else:
            guest.write(name+'\n')
