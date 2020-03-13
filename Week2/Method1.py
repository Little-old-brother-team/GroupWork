import numpy as np
class date:
    
    month_len = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype = np.int32)
    weekday_dict = {0:'Monday',1:'Tuesday',2:'Wedesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
    month_name = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',\
                  8:'August',9:'September',10:'Octomber',11:'November',12:'December'}
    
    def __init__(self, year , month, day):
        self.year = year
        self.month = month
        self.day = day
        
    def print_today(self):
        self.NumberWhich()
        print(f'{self.month_name[self.month]:<10s} {self.day:<3d}, {self.year:<20}{self.weekday:s}')

    def leap_years_before_today(self):
        leap_year = np.arange(1900 , self.year + 1, 4)
        special_years = np.arange(2000,self.year+1,400)
        self.leap_year = leap_year[leap_year % 100 != 0]
        self.leap_year = np.append(self.leap_year , special_years)
        self.leap_year.dtype = np.int32
        
    def how_many_days_passed(self):
        self.leap_years_before_today()
        days = 365 * ( self.year - 1900 ) + self.leap_year.__len__()
        if self.year in self.leap_year:
            days = days - 1
        days = days + np.sum(self.month_len[0:self.month-1]) + self.day - 1
        if self.year in self.leap_year and self.month>2:
            days = days + 1
        return days
    
    def NumberWhich(self):
        days = self.how_many_days_passed()
        no = days % 7
        self.weekday = self.weekday_dict[no]
            
if __name__ == '__main__':

    days = np.array([[1900,1,1],[1919,6,28],[1928,1,30],[1933,12,5],[1948,2,29],\
        [1948,3,1],[1953,1,15],[1963,11,22],[1993,6,23],[2005,8,28],[2111,5,16]])
    for i in range(days.__len__()):
        day = date(days[i,0],days[i,1],days[i,2])
        day.print_today()
    
