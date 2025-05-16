import datetime

def calculate_days():
    global Y
    global M
    global D

    date = datetime.date.today()

    days = 0

    #calcualting the days lived in the birth month
    if M in [1, 3, 5, 7, 8, 10, 12]:
        days = 31 - D
    elif M == 2:
        if (Y/4 == 0 and Y/100 != 0) or Y/400 == 0:
            days = 29 - D
        else:
            days = 28 - D
    elif M in [4, 6, 8, 9, 11]:
        days = 30 - D

    #adding the days lived in the birth year
    for i in range(M+1, 13):
        if i in [1, 3, 5, 7, 8, 10, 12]:
            days += 31
            
        elif i == 2:
            if (Y/4 == 0 and Y/100 != 0) or Y/400 == 0:
                days += 29
            else:
                days += 28

        else: days += 30

    #adding the days lived through the year after birth to previous year
    for i in range(Y+1, date.year):
        if i/4 == 0 and i/100 != 0 and i/400 == 0:
            days += 366
        else:
            days += 365

    #adding the days lived in the current year till previous month end
    for i in range(1, date.month):
        if i in [1, 3, 5, 7, 8, 10, 12]:
            days += 31
        elif i == 2:
            if (i/4 == 0 and i/100 != 0) or i/400 == 0:
                days += 29
            else:
                days += 28
        else:
            days += 30
    #adding the days lived in the current month till date
    days += date.day

    return days
    


#main
print("Enter your D.O.B. -")
Y = int(input("Enter year: "))
M = int(input("Enter month: "))
D = int(input("Enter day: "))

d = calculate_days()
years = d//365
days_rem = d%365

print("You are", years, "years,", days_rem, "days old")
