from datetime import datetime


def whatday(Year, Month, Day):
    # Calculate day gap
    checkDay = datetime(1900, 1, 1)
    targetDay = datetime(Year, Month, Day)
    Days = (targetDay - checkDay).days

    weekDay = (Days+1) % 7

    if weekDay == 1:
        daystr = 'Monday'
    elif weekDay == 2:
        daystr = 'Tuesday'
    elif weekDay == 3:
        daystr = 'Wednesday'
    elif weekDay == 4:
        daystr = 'Thursday'
    elif weekDay == 5:
        daystr = 'Friday'
    elif weekDay == 6:
        daystr = 'Saturday'
    elif weekDay == 0:
        daystr = 'Sunday'            
            
    return daystr


print(whatday(1993,6,23))