while True:
    a = input('enter a number:')
    b = input('enter a number:')
    try:
        a = int(a)
        b = int(b)
    except ValueError:
        print("your enter isn't a number!")
    else:
        print(a + b)
    opt = input('quit?[y/N]')

    if opt == 'y':
        break