
D = {'M':1000, 'CM':900, 'D':500, 'CD':400, 'C':100, 'XC':90,
     'L':50, 'XL':40, 'X':10, 'IX':9, 'V':5, 'IV':4, 'I':1}

n = int(input("Enter integer: "))
s = str(n)

r = ''

for key in D:
    if n//D[key]:
        c = n//D[key]
        r += key*int(c)

        n %= D[key]

print(r)






