import numpy as np

weekday_dict = {0:'Monday',1:'Tuesday',2:'Wedesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
month_name = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',\
                  8:'August',9:'September',10:'Octomber',11:'November',12:'December'}

def weekday(year,month,day):
    date_str = str(year)+'-'+str(month).rjust(2,'0')+'-'+str(day).rjust(2,'0')
    day_start = np.datetime64('1900-01-01')
    today = np.datetime64(date_str)
    delta_day = (today - day_start).astype(int)
    weekday_number = delta_day % 7
    print(f'{month_name[month]:<10s} {day:<3d}, {year:<20}{weekday_dict[weekday_number]:s}')
    
days = np.array([[1900,1,1],[1919,6,28],[1928,1,30],[1933,12,5],[1948,2,29],\
    [1948,3,1],[1953,1,15],[1963,11,22],[1993,6,23],[2005,8,28],[2111,5,16]])

for i in range(len(days)):
    weekday(days[i,0],days[i,1],days[i,2])
