dic={
    'nile':'egypt',
    'yellow river':'shandong',
    'yangtze river':'shanghai',
    }
for river,province in dic.items():
    print(f"The {river} runs through {province}")
for river in sorted(dic.keys()):
    print(f"{river}")
for province in sorted(dic.values()):
    print(f"{province}")
