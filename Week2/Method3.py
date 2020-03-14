import numpy as np

weekdayList = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
date_list = ['1919-06-28', '1928-01-30', '1933-12-05', '1948-02-29', '1948-03-01', '1953-01-15', '1963-11-22', '1993-06-23', '2005-08-28', '2111-05-16']

def getWeekday(datetime_str):
    std_date = np.datetime64('1900-01-01')
    target_date = np.datetime64(datetime_str)
    delta = target_date - std_date
    weekday = (delta % np.timedelta64(7, 'D')) / np.timedelta64(1, 'D')
    return weekdayList[int(weekday)]

for date in date_list:
    print(date + '\t' + getWeekday(date))
