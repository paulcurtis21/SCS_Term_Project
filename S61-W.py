import numpy as np
import matplotlib.pyplot as plt

def S61W(F, T, sigma, alpha, w):
    """
    T is nondimensional temperature difference; F is the nondimensional flux.
    sigma, alpha and w are parameters which need to be specified at the start.
    The sign of the flow always needs to be assessed.
    """
    
    # Fixed Points
    parameter_1 = alpha*T**2 + (w - 1)*T + 1
    parameter_sign_1 = np.sign(parameter_1)

    parameter_2 = alpha*T**2 + (w + 1)*T - 1
    parameter_sign_2 = np.sign(parameter_2)
    
    FPs = []
    for i in range(len(T) - 1):
        if np.abs(parameter_sign_1[i] - parameter_sign_1[i + 1]) == 2: # >0 
            T_FP = T[i] 
            F_FP = w + alpha*T_FP
            FP = [F_FP, T_FP]
            FPs = np.append(FPs, FP)
        elif np.abs(parameter_sign_2[i] - parameter_sign_2[i + 1]) == 2: # <0
            T_FP = T[i] 
            F_FP = w + alpha*T_FP
            FP = [F_FP, T_FP]
            FPs = np.append(FPs, FP)
    
    # Define the right hand side of the two time derivatives
    def f(F, T, sigma, alpha, w): # F dot
        f = sigma*(alpha*T + w - F)
        return f
    
    def g(F, T): # T dot
        g = 1 - T - np.abs(F)*T
        return g
    
    def Jacobian(F_FP, _FP): 
        delta_F = F[1] - F[0]
        delta_T = T[1] - T[0]
        
        a = (f((F_FP + 0.5*delta_F), T_FP, sigma, alpha, w) - f((F_FP - 0.5*delta_F), T_FP, sigma, alpha, w))/delta_F
        b = (f(F_FP, (T_FP + 0.5*delta_T), sigma, alpha, w) - f(F_FP, (T_FP - 0.5*delta_T), sigma, alpha, w))/delta_T
        c = (g((F_FP + 0.5*delta_F), T_FP) - g((F_FP - 0.5*delta_F), T_FP))/delta_F
        d = (g(F_FP, (T_FP + 0.5*delta_F)) - g(F_FP, (T_FP - 0.5*delta_F)))/delta_T
        
        determinant = a*d - b*c
        trace = a + d
        
        return [trace, determinant]
    
    for i in range(int(len(FPs)/2)):
        stability = Jacobian(FPs[2*i], FPs[2*i + 1])
        
        if stability[0] > 0 and stability[1] > 0 and stability[1] > 0.25*stability[0]**2:
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is an unstable spiral")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
        if stability[0] < 0 and stability[1] > 0 and stability[1] > 0.25*stability[0]**2:
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is a stable spiral")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'green')
            
        if stability[0] > 0 and stability[1] > 0 and stability[1] < 0.25*stability[0]**2: 
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is an unstable node")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
        if stability[0] < 0 and stability[1] > 0 and stability[1] < 0.25*stability[0]**2: 
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is a stable node")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'blue')
        
        if stability[0] > 0 and stability[1] < 0:
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is an unstable node")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
        if stability[0] < 0 and stability[1] < 0:
            print("x = ", FPs[2*i], "y = ", FPs[2*i + 1], "is an unstable node")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
    
    # Plot 
    plt.xlim([min(F), max(F)])
    plt.ylim([min(T), max(T)])
    
    plt.xlabel("Flow Rate (F)", fontsize = 25)
    plt.ylabel("Temperature (T)", fontsize = 25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #plt.title("Stommel 61 Model with Wind Driven correction", fontsize = 20)
    
    
    def RK4_S61W(sigma, alpha, w, t_max, n, F_0, T_0):
    t_step = t_max/n
    t = np.arange(0, t_max, t_step)
    F = np.zeros(int(n))
    T = np.zeros(int(n))
    [F[0], T[0]] = [F_0, T_0]
    
    def f(F, T, sigma, alpha, w): # F dot
        f = sigma*(alpha*T + w - F)
        return f
    
    def g(F, T): # T dot
        g = -1*np.abs(F)*T + 1 - T
        return g
    
    for i in range(len(t) - 1):
        f1 = f(F[i], T[i], sigma, alpha, w)
        g1 = g(F[i], T[i])

        f2 = f(F[i] + f1*t_step/2, T[i] + f1*t_step/2, sigma, alpha, w)
        g2 = g(F[i] + f1*t_step/2, T[i] + f1*t_step/2)
        
        f3 = f(F[i] + f2*t_step/2, T[i] + f2*t_step/2, sigma, alpha, w)
        g3 = g(F[i] + f2*t_step/2, T[i] + f2*t_step/2)
        
        f4 = f(F[i] + f3*t_step, T[i] + f3*t_step, sigma, alpha, w)
        g4 = g(F[i] + f3*t_step, T[i] + f3*t_step)

        F[i + 1] = F[i] + t_step*(1/6)*(f1 + 2*f2 + 2*f3 + f4)
        T[i + 1] = T[i] + t_step*(1/6)*(g1 + 2*g2 + 2*g3 + g4)
        
    return t, F, T

  
F = np.linspace(-3, 1, 2000000)
T = np.linspace(0, 1, 1000000)
sigma = 1
alpha = 4
w = -2
plt.rcParams['figure.figsize'] = [10, 10]


F_0 = [-3, -3, -3, -2, -1, 0, 3, 3, 3, 0, -2, -1, 0, 1, 1]
T_0 = [0, 0.5, 1, 1, 1, 1, 1, 0.5, 0, 0, 0, 0, 0, 0, 0.5]
for i in range(len(F_0)):
    [t, F_, T_] = RK4_S61W(sigma, alpha, w, 100, 10000, F_0[i], T_0[i])
    plt.plot(F_, T_, color = 'black', linewidth = 2)
plt.text(-2.8, 0.92, 'a)', fontsize = 40, weight='bold')
S61W(F, T, sigma, alpha, w)
plt.show()
