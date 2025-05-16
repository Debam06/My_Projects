nums1 = []
nums2 = []

while True:
    try:
        val = float(input("Enter value for first array(press any letter to stop): "))
        nums1 += [val]
    except:
        break

while True:
    try:
        val = float(input("Enter value for second array(press any letter to stop): "))
        nums2 += [val]
    except:
        break

L = nums1 + nums2
L.sort()

k = len(L)

if k%2 == 0:
    median = (L[k//2] + L[k//2 - 1])/2
    print(median)
else:
    median = L[k//2]
    print(median)
