import numpy as np 
def getSolution1(parameter):
    a, b, c = parameter
    delta = np.sqrt(b**2 - 4*a*c)
    return (-b + delta)/(2*a), (-b - delta)/(2*a)

def getSolution2(parameter):
    a, b, c = parameter
    delta = np.sqrt(b**2 - 4*a*c)
    return 2*c/(-b + delta), 2*c/(-b - delta)

def getPresiceSolution(parameter):
    a, b, c = parameter
    delta = np.sqrt(b**2 - 4*a*c)
    if b >= 0:
        dom = -b - delta
        return dom / 2 / a, 2 * c / dom
    else:
        dom = -b + delta
        return dom / 2 / a, 2 * c / dom

if __name__ == '__main__':
    print(getSolution1([1, -10000.0001, 1]))
    print(getSolution2([1, -10000.0001, 1]))
    print(getPresiceSolution([1, -10000.0001, 1]))
    print("hello")