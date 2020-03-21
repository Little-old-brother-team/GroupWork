# 牛顿法
def newton_method(quad , x_temp , x_last = float('inf')):
    if quad[1]**2 - 4 * quad[0] * quad[2] < 0:         #判断是否有根
        return False
    
    quad[1] = quad[1] / quad[0]
    quad[2] = quad[2] / quad[0]     #标准化二次函数的系数
    quad[0] = 1                     
    
    tolerance = 1e-19
    
    if abs (( x_temp - x_last ) / x_temp ) < tolerance:     #若满足精度，就返回x_temp
        return x_temp
    else:
        x_last = x_temp
        x_temp = ( x_temp ** 2 - quad[2] ) / ( 2 * x_temp + quad[1] )   #不满足误差要求则将x0处切线的零点带入
        return newton_method(quad,x_temp,x_last)

quad = [1,-1000.001,1]   #二次函数的系数
print(f'x_1 = {newton_method([1,-1000.001,1],-100):.50f}\nx_2 = {newton_method([1,-1000.001,1],1000):.47f}')