album = {}
def make_album(name, song_name, num=None):
    album['name'] = name
    album['song_name'] = song_name
    if num:
        album['num'] = num

while True:
    name = input("the singer's name is: \n")
    song_name = input("the song's name is: \n")
    make_album(name, song_name)
    if input('do you want to quit? [y/N]\n') == 'y':
        break

print(album)