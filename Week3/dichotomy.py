def dichotomy(a,b,c):
    # In this task, a = c = 1, b>>1. Thus one root near 0 while the other near -b.
    def find_sol(left,right):
        while abs(right-left) >= 0.00000000000000001: 
            mid = (right+left)/2
            l = binomial(left)
            r = binomial(right)
            if l == 0:
                sol = left
                break
            elif r == 0:
                sol = right
                break
            m = binomial(mid)
        
            if l*r > 0:
                sol = 'No solution'
                break
            elif m == 0:
                sol = mid
                break
            elif m*l < 0:
                right = mid
            elif m*r < 0:
                left = mid
                sol = mid
        return sol

    def binomial(x):
        y = a*x**2 + b*x + c
        return(y)
    
    sol_left = find_sol(left=-abs(b),right=abs(b/2))
    sol_right = find_sol(right=abs(b),left=abs(b/2))
    return sol_left,sol_right
    
print(dichotomy(1,-1000.001,1))