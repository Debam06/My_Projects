def reverse_num(n):

    if n > 0:
        s = str(n)
        L = []
        for c in s:
            L += [c]
        L.reverse()

        s1 = ""
        for c in L:
            s1 += c
        reversed_num = int(s1)

        return reversed_num
        
    elif n < 0:
        s = str(n)
        L = []
        for i in range(1, len(s)):
            L += s[i]
        L.reverse()

        s1 = ""
        for c in L:
            s1 += c
        reversed_num = int(s1)

        return reversed_num*(-1)


#main
n = int(input("Enter number: "))
n1 = reverse_num(n)

print(n1)
        
