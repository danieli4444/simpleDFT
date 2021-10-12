import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from scipy import integrate


def runNumerov(g,f0,df,h):
    """ Numerov algortihm:
    f[n+1] = [ 2*f[n]*(1 - 5*h**2/12 * g[n]) - f[n-1]*(1 + h**2/12 * g[n-1]) + h**2/12 *( s[n+1] + 10*s[n] + s[n-1] ) ] / ( 1+ h**2/12 * g[n+1])

    returns f(n)

    Args:
        g ([type]): function for which we solve the DE - f''(x)  + g(x)f(x) = s(x) ; in our case s(x) = 0.
        f0 ([type]): f(0)
        df ([type]): delta_f (used for estimating f[1])
        h ([type]): grid spacing
    """
    s = np.zeros(len(g))
    f = np.zeros(len(g))
    f[0] = f0
    f[1] = f0+ df*h
    for n in range(2,len(g)-1):
        w0 = f[n-2] * ( 1 - (h**2/12) * g[n-2]) - h**2/12 * s[n-2]
        w1 = f[n-1] * ( 1 - (h**2/12) * g[n-1]) - h**2/12 * s[n-1]
        w2 = 2*w1 - w0 + h**2 * g[n-1]*f[n-1]
        f[n] = w2/(1 - h**2/12 *g[n])
    return f

def Numerov(f, x0, dx, dh):
    """Given precomputed function f(x), solves for x(t), which satisfies:
          x''(t) = f(t) x(t)
    """
    x = np.zeros(len(f))
    x[0] = x0
    x[1] = x0+dh*dx
    print(dh)
    print(x[1])
    h2 = dh**2
    h12 = h2/12.

    w0=x0*(1-h12*f[0])
    w1=x[1]*(1-h12*f[1])
    xi = x[1]
    fi = f[1]
    for i in range(2,len(f)):
        w2 = 2*w1-w0+h2*fi*xi  # here fi=f1
        fi = f[i]  # fi=f2
        xi = w2/(1-h12*fi)
        x[i]=xi
        w0 = w1
        w1 = w2
    print(x[0])
    print("fuck")
    return x


def fSchrod(En, l, R):
    return l*(l+1.)/R**2-2./R-En


def test_Numerov2():
    Rl = linspace(1e-7,50,1000)
    l=0
    En=-1.
    f = fSchrod(En,l,Rl[::-1])
    ur = runNumerov(f,0.0,1e-7,Rl[1]-Rl[0])[::-1]
    print(ur[0],ur[1],ur[2],ur[3])
    print(Rl[0],Rl[1],Rl[2],Rl[3])
    norm = integrate.simps(ur**2,x=Rl)
    ur *= 1/np.sqrt(abs(norm))
    plt.xlabel('x')
    plt.ylabel('V')
    plt.title('V=x^2') 
    plt.plot(Rl , ur , color='blue')
    plt.show()


def test_Numerov():
    """ d2(V) = g(x) where g(x) = 1,1,1,1,
        V = x^2
    """
    l = 1000
    g = np.ones(l)
    (Rvec, grid_dr) = np.linspace(0, 1000, l,endpoint=False,retstep=True)

    V = Numerov(g,0,2,grid_dr)
    print(V[0],V[1],V[2],V[3])
    print(Rvec[0],Rvec[1],Rvec[2],Rvec[3])
    plt.xlabel('x')
    plt.ylabel('V')
    plt.title('V=x^2') 
    plt.plot(Rvec , V , color='blue')
    plt.show()

test_Numerov()
