nums = []
L = []

while True:
    try:
        n = int(input("Enter integer (press any letter key to stop): "))
        nums += [n]
    except:
        break

target = int(input("Enter target: "))

for i in range(len(nums)):
    n1 = nums[i]
    for j in range(len(nums)):
        n2 = nums[j]
        if i != j:
            if target == n1 + n2:
                if i not in L:
                    L += [i, j]

print(L)
