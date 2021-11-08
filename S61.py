import numpy as np
import matplotlib.pyplot as plt

#%%

def S61(y, x, R, delta, lam):
    """
    y is nondimensional temperature; x is nondimensional salinity.
    R, delta and lambda are parameters which need to be specified at the start.
    The sign of the flow always needs to be assessed.
    """
    
    # Fixed Points
    parameter_1 = (1 - delta)*y**3 + (delta*(R + lam) - (1 + lam))*y**2 + lam*(2 - delta)*y - lam
    parameter_sign_1 = np.sign(parameter_1)

    parameter_2 = (delta - 1)*y**3 + (delta*(lam - R) + (1 - lam))*y**2 + lam*(2 - delta)*y - lam
    parameter_sign_2 = np.sign(parameter_2)
    
    FPs = []
    for i in range(len(y) - 1):
        if abs(parameter_sign_1[i] - parameter_sign_1[i + 1]) == 2: # >0 
            y_FP = y[i] 
            x_FP = (lam/(R*y_FP))*(1 - y_FP + (1/lam)*y_FP**2)
            FP = [x_FP, y_FP]
            FPs = np.append(FPs, FP)
        elif abs(parameter_sign_2[i] - parameter_sign_2[i + 1]) == 2: # <0
            y_FP = y[i] 
            x_FP = (lam/(R*y_FP))*(-1 + y_FP + (1/lam)*y_FP**2)
            FP = [x_FP, y_FP]
            FPs = np.append(FPs, FP)
            
    # Flow rate at fixed points
    f_ = np.zeros(int(len(FPs)/2))
    for i in range(int(len(FPs)/2)):
        f_[i] = (R*FPs[2*i] - FPs[2*i + 1])/lam
    
    # Define the right hand side of the two time derivatives
    def f(y, x, lam, R, delta): # x dot
        f = delta*(1 - x) - (x/lam)*np.abs(-y + R*x)
        return f
    
    def g(y, x, lam, R): # y dot
        g = 1 - y -(y/lam)*np.abs(-y + R*x)
        return g
    
    # Jacobian analysis for stationary points
    def Jacobian(x_FP, y_FP): 
        delta_y = y[1] - y[0]
        delta_x = x[1] - x[0]
        
        a = (f(y_FP, (x_FP  + 0.5*delta_x), lam, R, delta) - f(y_FP, (x_FP  - 0.5*delta_x), lam, R, delta))/delta_x
        b = (f((y_FP + 0.5*delta_y), x_FP, lam, R, delta) - f((y_FP - 0.5*delta_y), x_FP, lam, R, delta))/delta_y
        c = (g(y_FP, (x_FP  + 0.5*delta_x), lam, R) - g(y_FP, (x_FP  - 0.5*delta_x), lam, R))/delta_x
        d = (g((y_FP + 0.5*delta_y), x_FP, lam, R) - g((y_FP - 0.5*delta_y), x_FP, lam, R))/delta_y
        
        determinant = a*d - b*c
        trace = a + d
        
        return [trace, determinant]
    
    stability = np.zeros(int(len(FPs)/2))
    for i in range(int(len(FPs)/2)):
        stability = Jacobian(FPs[2*i], FPs[2*i + 1])
        
        if stability[0] > 0 and stability[1] > 0 and stability[1] > 0.25*stability[0]**2:
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is an unstable spiral. f = ", f_[i])
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
        if stability[0] < 0 and stability[1] > 0 and stability[1] > 0.25*stability[0]**2:
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is a stable spiral. f = ", f_[i])
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'blue')
            
        if stability[0] > 0 and stability[1] > 0 and stability[1] < 0.25*stability[0]**2: 
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is an unstable node. f = ", f_[i])
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
        if stability[0] < 0 and stability[1] > 0 and stability[1] < 0.25*stability[0]**2: 
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is a stable node. f = ", f_[i])
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'blue')
        
        if stability[0] > 0 and stability[1] < 0:
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is an unstable node. f = ", f_[i])
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
        if stability[0] < 0 and stability[1] < 0:
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is an unstable node. f = ", f_[i])
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
    
    # Plot 
    plt.xlim([min(x), max(x)])
    plt.ylim([min(y), max(y)])
    
    plt.xlabel("x", fontsize = 20)
    plt.ylabel("y", fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Stommel 61 Model", fontsize = 20)
    
    # Plot the lines of constant flow
    n = np.arange(-4, 10, 2)
    for i in range(len(n)):
        f__ = R*x - n[i]*lam
        plt.plot(x, f__, '--', color = 'black')

    plt.show()
    
#%%

def RK4_S61(R, delta, lam, t_max, n, x_0, y_0):
    t_step = t_max/n
    t = np.arange(0, t_max, t_step)
    x = np.zeros(int(n))
    y = np.zeros(int(n))
    [x[0], y[0]] = [x_0, y_0]
    
    def f(x, y, lam, R, delta): # x dot
        f = delta*(1 - x) - (x/lam)*np.abs(-y + R*x)
        return f
    
    def g(x, y, lam, R): # y dot
        g = 1 - y -(y/lam)*np.abs(-y + R*x)
        return g
    
    for i in range(len(t) - 1):
        f1 = f(x[i], y[i], lam, R, delta)
        g1 = g(x[i], y[i], lam, R)

        f2 = f(x[i] + f1*t_step/2, y[i] + g1*t_step/2, lam, R, delta)
        g2 = g(x[i] + f1*t_step/2, y[i] + g1*t_step/2, lam, R)

        f3 = f(x[i] + f2*t_step/2, y[i] + g2*t_step/2, lam, R, delta)
        g3 = g(x[i] + f2*t_step/2, y[i] + g2*t_step/2, lam, R)

        f4 = f(x[i] + f3*t_step, y[i] + g3*t_step, lam, R, delta)
        g4 = g(x[i] + f3*t_step, y[i] + g3*t_step, lam, R)

        x[i + 1] = x[i] + t_step*(1/6)*(f1 + 2*f2 + 2*f3 + f4)
        y[i + 1] = y[i] + t_step*(1/6)*(g1 + 2*g2 + 2*g3 + g4)
        
    return t, x, y

#%% 

y = np.linspace(0, 1, 1000000)
x = np.linspace(0, 1, 1000000)
R = 2
delta = 1/6
lam = 1/5
plt.rcParams['figure.figsize'] = [10, 10]


x_0 = [0, 0.2, 1, 1, 1, 0.4, 0, 0]
y_0 = [0, 0, 0.1, 0.5, 1, 1, 0.75, 0.4]
for i in range(8):
    [t, x_, y_] = RK4_S61(2, 1/6, 1/5, 100, 10000, x_0[i], y_0[i])
    plt.plot(x_, y_, color = 'black', linewidth = 2)
S61(y, x, R, delta, lam)
