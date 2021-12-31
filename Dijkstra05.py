import numpy as np
import matplotlib.pyplot as plt

#%%

def S61D(T, S, eta1, eta2, eta3):
    """
    T is nondimensional temperature; S is nondimensional salinity.
    eta1, eta2 and eta3 are parameters which need to be specified at the start.
    The sign of the flow always needs to be assessed.
    """
    
    # Fixed Points
    parameter_1 = (1 - eta3)*T**3 + (eta2 - eta3 - eta1 + 1)*T**2 + (eta1*eta3 - 2*eta1)*T + eta1**2
    parameter_sign_1 = np.sign(parameter_1)

    parameter_2 = (1 - eta3)*T**3 + (eta2 + eta3 - eta1 - 1)*T**2 + (-eta1*eta3 + 2*eta1)*T - eta1**2
    parameter_sign_2 = np.sign(parameter_2)
    
    FPs = []
    for i in range(len(T) - 1):
        if abs(parameter_sign_1[i] - parameter_sign_1[i + 1]) == 2: # >0 
            T_FP = T[i] 
            S_FP = (1/T_FP)*(T_FP**2 + T_FP - eta1)
            FP = [T_FP, S_FP]
            FPs = np.append(FPs, FP)
        elif abs(parameter_sign_2[i] - parameter_sign_2[i + 1]) == 2: # <0
            T_FP = T[i] 
            S_FP = (1/T_FP)*(T_FP**2 - T_FP + eta1)
            FP = [T_FP, S_FP]
            FPs = np.append(FPs, FP)
    print(FPs)
            
    def f(T, S, eta1): # x dot
        f = eta1 - T*(1 + np.abs(T - S))
        return f
    
    def g(T, S, eta2, eta3): # y dot
        g = eta2 - S*(eta3 + np.abs(T - S))
        return g
    
    def Jacobian(T_FP, S_FP): 
        delta_T = T[1] - T[0]
        delta_S = S[1] - S[0]
        
        a = (f((T_FP + 0.5*delta_T), S_FP, eta1) - f((T_FP - 0.5*delta_T), S_FP, eta1))/delta_T
        b = (f(T_FP, (S_FP + 0.5*delta_S), eta1) - f(T_FP, (S_FP - 0.5*delta_S), eta1))/delta_S
        c = (g((T_FP + 0.5*delta_T), S_FP, eta2, eta3) - g((T_FP - 0.5*delta_T), S_FP, eta2, eta3))/delta_T
        d = (g(T_FP, (S_FP + 0.5*delta_S), eta2, eta3) - g(T_FP, (S_FP - 0.5*delta_S), eta2, eta3))/delta_S
        
        determinant = a*d - b*c
        trace = a + d
        
        return [trace, determinant]
    
    stability = np.zeros(int(len(FPs)/2))
    for i in range(int(len(FPs)/2)):
        stability = Jacobian(FPs[2*i], FPs[2*i + 1])
        print(stability)
        
        if stability[0] > 0 and stability[1] > 0 and stability[1] > 0.25*stability[0]**2:
            print("T = ", FPs[2*i], "S = ", FPs[2*i + 1], "is an unstable spiral. f = ")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
        if stability[0] < 0 and stability[1] > 0 and stability[1] > 0.25*stability[0]**2:
            print("T = ", FPs[2*i], "S = ", FPs[2*i + 1], "is a stable spiral. f = ")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'blue')
            
        if stability[0] > 0 and stability[1] > 0 and stability[1] < 0.25*stability[0]**2: 
            print("T = ", FPs[2*i], "S = ", FPs[2*i + 1], "is an unstable node. f = ")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
        if stability[0] < 0 and stability[1] > 0 and stability[1] < 0.25*stability[0]**2: 
            print("T = ", FPs[2*i], "S = ", FPs[2*i + 1], "is a stable node. f = ")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'blue')
        
        if stability[0] > 0 and stability[1] < 0:
            print("T = ", FPs[2*i], "S = ", FPs[2*i + 1], "is an unstable node. f = ")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
        if stability[0] < 0 and stability[1] < 0:
            print("T = ", FPs[2*i], "S = ", FPs[2*i + 1], "is an unstable node. f = ")
            plt.plot(FPs[2*i], FPs[2*i + 1], markersize = 14, marker = 'o', color = 'red')
            
    plt.xlim([min(T), max(T)])
    plt.ylim([min(S), max(S)])
    
    plt.xlabel("T", fontsize = 20)
    plt.ylabel("S", fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Stommel 61 Model", fontsize = 20)

    plt.show()
    
#%%
# This is just a test to make sure the code is working

eta1 = 3
eta2 = 1
eta3 = 0.3
T = np.linspace(0, 4, 1000000)
S = np.linspace(0, 4, 1000000)
S61D(T, S, eta1, eta2, eta3)

#%% 
# Code to produce the bifurcation diagrams

def S61DB(T, S, eta1, eta3, eta2_min, eta2_max, eta2_vals):
    """
    T is nondimensional temperature; S is nondimensional salinity.
    eta1, eta2 and eta3 are parameters which need to be specified at the start.
    The sign of the flow always needs to be assessed.
    """
    
    eta2 = np.linspace(eta2_min, eta2_max, eta2_vals)
    
    F_stable_node = np.zeros(len(eta2))
    F_stable_spiral = np.zeros(len(eta2))
    F_unstable_node = np.zeros(len(eta2))
    
    def f(T, S, eta1): # x dot
        f = eta1 - T*(1 + np.abs(T - S))
        return f
    
    def g(T, S, eta2, eta3): # y dot
        g = eta2 - S*(eta3 + np.abs(T - S))
        return g
    
    def phi(T, S):
        phi = T - S
        return phi
    
    def Jacobian(T_FP, S_FP, eta2): 
        delta_T = T[1] - T[0]
        delta_S = S[1] - S[0]
        
        a = (f((T_FP + 0.5*delta_T), S_FP, eta1) - f((T_FP - 0.5*delta_T), S_FP, eta1))/delta_T
        b = (f(T_FP, (S_FP + 0.5*delta_S), eta1) - f(T_FP, (S_FP - 0.5*delta_S), eta1))/delta_S
        c = (g((T_FP + 0.5*delta_T), S_FP, eta2, eta3) - g((T_FP - 0.5*delta_T), S_FP, eta2, eta3))/delta_T
        d = (g(T_FP, (S_FP + 0.5*delta_S), eta2, eta3) - g(T_FP, (S_FP - 0.5*delta_S), eta2, eta3))/delta_S
        
        determinant = a*d - b*c
        trace = a + d
        
        return [trace, determinant]
    
    for j in range(len(eta2)):
    
        # Fixed Points
        parameter_1 = (1 - eta3)*T**3 + (eta2[j] - eta3 - eta1 + 1)*T**2 + (eta1*eta3 - 2*eta1)*T + eta1**2
        parameter_sign_1 = np.sign(parameter_1)

        parameter_2 = (1 - eta3)*T**3 + (eta2[j] + eta3 - eta1 - 1)*T**2 + (-eta1*eta3 + 2*eta1)*T - eta1**2
        parameter_sign_2 = np.sign(parameter_2)

        FPs = []
        for i in range(len(T) - 1):
            if abs(parameter_sign_1[i] - parameter_sign_1[i + 1]) == 2: # >0 
                T_FP = T[i] 
                S_FP = (1/T_FP)*(T_FP**2 + T_FP - eta1)
                FP = [T_FP, S_FP]
                FPs = np.append(FPs, FP)
            elif abs(parameter_sign_2[i] - parameter_sign_2[i + 1]) == 2: # <0
                T_FP = T[i] 
                S_FP = (1/T_FP)*(T_FP**2 - T_FP + eta1)
                FP = [T_FP, S_FP]
                FPs = np.append(FPs, FP)
    
        for i in range(int(len(FPs)/2)):
                stability = Jacobian(FPs[2*i], FPs[2*i + 1], eta2[j])

                # unstable spiral
                if stability[0] > 0 and stability[1] > 0 and stability[1] > 0.25*stability[0]**2:
                    F_stable_node[j] = FPs[2*i] - FPs[2*i + 1]
                # stable spiral
                if stability[0] < 0 and stability[1] > 0 and stability[1] > 0.25*stability[0]**2:
                    F_stable_spiral[j] = FPs[2*i] - FPs[2*i + 1]
                # unstable node
                if stability[0] > 0 and stability[1] > 0 and stability[1] < 0.25*stability[0]**2: 
                    F_unstable_node[j] = FPs[2*i] - FPs[2*i + 1]
                # stable node
                if stability[0] < 0 and stability[1] > 0 and stability[1] < 0.25*stability[0]**2: 
                    F_stable_node[j] = FPs[2*i] - FPs[2*i + 1]
                # unstable node
                if stability[0] > 0 and stability[1] < 0:
                    F_unstable_node[j] = FPs[2*i] - FPs[2*i + 1]
                # unstable node
                if stability[0] < 0 and stability[1] < 0:
                    F_unstable_node[j] = FPs[2*i] - FPs[2*i + 1]

    return eta2, F_stable_node, F_stable_spiral, F_unstable_node

#%%

T = np.linspace(0.5, 3, 10000)
S = np.linspace(0.5, 3, 10000)
eta1 = 3
eta3 = 0.3
eta2_min = 0
eta2_max = 2
eta2_vals = 5000

[eta1, F_stable_node, F_stable_spiral, F_unstable_node] = S61DB(T, S, eta1, eta3, eta2_min, eta2_max, eta2_vals)

for i in range(int(len(eta1))):
    if round(F_stable_node[i], 15) == 0:
        F_stable_node[i] = np.nan
    if round(F_stable_spiral[i], 15) == 0:
        F_stable_spiral[i] = np.nan
    if round(F_unstable_node[i], 15) == 0:
        F_unstable_node[i] = np.nan
        
 plt.rcParams['figure.figsize'] = [10, 10]
plt.plot(eta1, F_stable_node, linewidth = 4, color = 'blue')
plt.plot(eta1, F_stable_spiral, linewidth = 4, color = 'green')
plt.plot(eta1, F_unstable_node, linewidth = 4, color = 'red')
plt.axhline(0, linestyle = '--', color = 'black')

# Add other features to the bifurcation diagram
plt.axvline(1.22, ymin = 0.29, ymax = 0.47, linewidth = 3, color = 'black')
plt.axvline(0.9, ymin = 0.333, ymax = 0.61, linewidth = 3, color = 'black')
plt.arrow(1.221, 0.17, 0, -0.1, length_includes_head = True, head_width = 0.05, head_length= 0.05, color = 'black')
plt.arrow(0.903, 0.3, 0, 0.2, length_includes_head = True, head_width = 0.05, head_length= 0.05, color = 'black')

plt.xlabel("$\eta_{2}", fontsize = 25)
plt.ylabel("Flow Rate (F)", fontsize = 25)
plt.xlim([0, 2])
plt.ylim([-1, 2])
plt.xticks(np.arange(0, 2.5, step=0.5), fontsize = 20)
plt.yticks(fontsize = 20)
plt.text(0.08, 1.7, 'a)', fontsize = 40, weight='bold')
plt.show()
