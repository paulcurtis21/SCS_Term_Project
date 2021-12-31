import numpy as np
import matplotlib.pyplot as plt

%%

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
    
    #%%
    
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

#%%
  
F = np.linspace(-3, 1, 2000000)
T = np.linspace(0, 1, 1000000)
sigma = 1
alpha = 4.6
w = -3.7
plt.rcParams['figure.figsize'] = [10, 10]


F_0 = [-3, -3, -3, -2, -1, 0, 3, 3, 3, 0, -2, -1, 0, 1, 1]
T_0 = [0, 0.5, 1, 1, 1, 1, 1, 0.5, 0, 0, 0, 0, 0, 0, 0.5]
for i in range(len(F_0)):
    [t, F_, T_] = RK4_S61W(sigma, alpha, w, 100, 10000, F_0[i], T_0[i])
    plt.plot(F_, T_, color = 'black', linewidth = 2)
plt.text(-2.8, 0.92, 'b)', fontsize = 40, weight='bold')
S61W(F, T, sigma, alpha, w)
plt.xticks(np.arange(-3, 2, 1), fontsize = 25)
plt.show()

#%%

sigma = 1
alpha = 4.6
w = -3.7
[t, F_1, T_1] = RK4_S61W(sigma, alpha, w, 100, 10000, 1, 0.5)
[t, F_2, T_2] = RK4_S61W(sigma, alpha, w, 100, 10000, 1, 0.3)

plt.rcParams['figure.figsize'] = [8, 8]
plt.plot(t, T_1, color = 'green', linewidth = 4, label = "$T_{0} = 0.5$")
plt.plot(t, T_2, color = 'blue', linewidth = 4, label = "$T_{0} = 0.3$")
plt.xlim([0, 10])
plt.ylim([0, 1.2])
plt.xlabel("Integration Time", fontsize = 25)
plt.ylabel("Temperature (T)", fontsize = 25)
plt.text(0.6, 1.07, 'b)', fontsize = 40, weight='bold')
plt.xticks(np.arange(0, 25, 5), fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(fontsize = 20)
plt.show()

#%%

[t, F_1, T_1] = RK4_S61W(1, 4, -2, 100, 10000, 1, 0.5)
[t, F_2, T_2] = RK4_S61W(0.1, 4, -2, 100, 10000, 1, 0.5)

plt.rcParams['figure.figsize'] = [8, 8]
plt.plot(t, F_1, color = 'black', linewidth = 4, label = "$\sigma = 1$")
plt.plot(t, F_2, color = 'red', linewidth = 4, label = "$\sigma = 0.1$")
plt.xlim([0, 20])
plt.ylim([0.4, 1])
plt.xlabel("Integration Time", fontsize = 25)
plt.ylabel("Flow Rate (F)", fontsize = 25)
plt.xticks(np.arange(0, 25, 5), fontsize = 20)
plt.yticks(fontsize = 25)
plt.legend(fontsize = 25)
plt.show()

#%%
# Function to produce bifurcation diagrams

def S61WB(F, T, sigma, alpha, w_vals):
    """
    T is nondimensional temperature difference; F is the nondimensional flux.
    sigma, alpha and w are parameters which need to be specified at the start.
    The sign of the flow always needs to be assessed.
    """
    
    w = np.linspace(-5, 0, w_vals)
    
    F_stable_node = np.zeros(len(w))
    F_stable_spiral = np.zeros(len(w))
    F_unstable_node = np.zeros(len(w))
    
    for j in range(len(w)):
    
        # Fixed Points
        parameter_1 = alpha*T**2 + (w[j] - 1)*T + 1
        parameter_sign_1 = np.sign(parameter_1)

        parameter_2 = alpha*T**2 + (w[j] + 1)*T - 1
        parameter_sign_2 = np.sign(parameter_2)

        FPs = []
        for i in range(len(T) - 1):
            if np.abs(parameter_sign_1[i] - parameter_sign_1[i + 1]) == 2: # >0 
                T_FP = T[i] 
                F_FP = w[j] + alpha*T_FP
                FP = [F_FP, T_FP]
                FPs = np.append(FPs, FP)
            elif np.abs(parameter_sign_2[i] - parameter_sign_2[i + 1]) == 2: # <0
                T_FP = T[i] 
                F_FP = w[j] + alpha*T_FP
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

            a = (f((F_FP + 0.5*delta_F), T_FP, sigma, alpha, w[j]) - f((F_FP - 0.5*delta_F), T_FP, sigma, alpha, w[j]))/delta_F
            b = (f(F_FP, (T_FP + 0.5*delta_T), sigma, alpha, w[j]) - f(F_FP, (T_FP - 0.5*delta_T), sigma, alpha, w[j]))/delta_T
            c = (g((F_FP + 0.5*delta_F), T_FP) - g((F_FP - 0.5*delta_F), T_FP))/delta_F
            d = (g(F_FP, (T_FP + 0.5*delta_F)) - g(F_FP, (T_FP - 0.5*delta_F)))/delta_T

            determinant = a*d - b*c
            trace = a + d

            return [trace, determinant]

        # Sets up the arrays to plot

        for i in range(int(len(FPs)/2)):
            stability = Jacobian(FPs[2*i], FPs[2*i + 1])

            # unstable spiral
            if stability[0] > 0 and stability[1] > 0 and stability[1] > 0.25*stability[0]**2:
                F_stable_node[j] = FPs[2*i]
            # stable spiral
            if stability[0] < 0 and stability[1] > 0 and stability[1] > 0.25*stability[0]**2:
                F_stable_spiral[j] = FPs[2*i]
            # unstable node
            if stability[0] > 0 and stability[1] > 0 and stability[1] < 0.25*stability[0]**2: 
                F_unstable_node[j] = FPs[2*i]
            # stable node
            if stability[0] < 0 and stability[1] > 0 and stability[1] < 0.25*stability[0]**2: 
                F_stable_node[j] = FPs[2*i]
            # unstable node
            if stability[0] > 0 and stability[1] < 0:
                F_unstable_node[j] = FPs[2*i]
            # unstable node
            if stability[0] < 0 and stability[1] < 0:
                F_unstable_node[j] = FPs[2*i]

    return w, F_stable_node, F_stable_spiral, F_unstable_node

#%% 

F = np.linspace(-2.5, 1, 50000)
T = np.linspace(0, 1, 20000)
sigma = 1
alpha = 3
w_vals = 10000

[w, F_stable_node, F_stable_spiral, F_unstable_node] = S61WB(F, T, sigma, alpha, w_vals)

for i in range(int(len(w))):
    if round(F_stable_node[i], 15) == 0:
        F_stable_node[i] = np.nan
    if round(F_stable_spiral[i], 15) == 0:
        F_stable_spiral[i] = np.nan
    if round(F_unstable_node[i], 15) == 0:
        F_unstable_node[i] = np.nan
        
plt.rcParams['figure.figsize'] = [8, 8]
plt.plot(w, F_stable_node, linewidth = 4, color = 'blue')
plt.plot(w, F_stable_spiral, linewidth = 4, color = 'green')
plt.plot(w, F_unstable_node, linewidth = 4, color = 'red')
plt.axhline(0, linestyle = '--', color = 'black')

# Add other features to the bifurcation diagram
plt.axvline(-3, ymin = 0.43, ymax = 0.71, linewidth = 4, color = 'black')
plt.axvline(-2.465, ymin = 0.6, ymax = 0.73, linewidth = 4, color = 'black')
plt.arrow(-3, -0.7, 0, -0.3, length_includes_head = True, head_width = 0.1, color = 'black')
plt.arrow(-2.465, -0.35, 0, +0.2, length_includes_head = True, head_width = 0.1, color = 'black')
plt.scatter(-1.27, 1.02, 1, c = 'black', marker = 'x', linewidth = 2)

plt.xlabel("Wind Stress ($\omega$)", fontsize = 25)
plt.ylabel("Flow Rate (F)", fontsize = 25)
plt.xlim([-5, 0])
plt.ylim([-5, 2])
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.text(-4.8, 1.2, 'a)', fontsize = 40, weight='bold')
plt.show()
