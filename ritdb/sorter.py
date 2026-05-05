from random import shuffle

players = ['jared', 'henry', 'connor', 'jake', 'sam', 'will', 'adam', 'peter', 'caleb', 'katie', 'maddie', 'francesca', 'eric', 'danny'] 

shuffle(players)

reader = players[0]
room1 = players[1:5]
room2 = players[5:9]
room3 = players[9:]

rooms = (room1, room2, room3)


print(f'Reader: {reader}')
for k in range(3):
    print(f'ROOM {k+1}: {rooms[k]}')
