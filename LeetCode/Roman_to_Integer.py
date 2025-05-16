
D = {'I':'1', 'V':'5', 'X':'10', 'L':'50', 'C':'100', 'D':'500', 'M':'1000'}
D1 = {'IV':'4', 'IX':'9', 'XL':'40', 'XC':'90', 'CD':'400', 'CM':'900'}

r = input("Enter Roman numeral: ")

s = 0

for key in D1:
    if key in r:
        s += int(D1[key])
        r = r.replace(key, "")

for c in r:
    s += int(D[c])

print(s)
