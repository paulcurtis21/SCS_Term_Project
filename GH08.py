import numpy as np
import matplotlib.pyplot as plt

#%%

def RK4_GH08(t_max, n, T_0, S_0, c, omega, p):
    t_step = t_max/n
    t = np.arange(0, t_max, t_step)
    T = np.zeros(int(n))
    S = np.zeros(int(n))
    psi_ = np.zeros(int(n))
    rho_TS = np.zeros(int(n))
    [T[0], S[0]] = [T_0, S_0]
    
    alpha = 0.184
    beta = 0.774
    rho_0 = 1.023e3
    
    T0 = 15
    S0 = 35
    
   # flow rate
    def psi(T, S, c): 
        psi = c*(rho_0*(alpha*T - beta*S))
        return psi
    
    psi_[0] = psi(T[0], S[0], c)
        
    # T dot = ...
    def f(T, S, c, omega, p):
        if psi(T, S, c) >= 0:
            f = -2*psi(T, S, c)*T + (p - 1 - 2*omega)*T + (p + 1)*T0
        else:
            f = 2*psi(T, S, c)*T + (-p - 1 - 2*omega)*T + (p + 1)*T0
        return f
        
    # S dot = ...
    def g(T, S, c, omega, p):
        if psi(T, S, c) >= 0:
            g = -2*psi(T, S, c)*S + (p - 2*omega)*S + 2*p*S0
        else:
            g = 2*psi(T, S, c)*S + (-p - 2*omega)*S + 2*p*S0
        return g
        
    
    for i in range(len(t) - 1):
        f1 = f(T[i], S[i], c, omega, p)
        g1 = g(T[i], S[i], c, omega, p)

        f2 = f(T[i] + f1*t_step/2, S[i] + f1*t_step/2, c, omega, p)
        g2 = g(T[i] + f1*t_step/2, S[i] + f1*t_step/2, c, omega, p)
        
        f3 = f(T[i] + f2*t_step/2, S[i] + f2*t_step/2, c, omega, p)
        g3 = g(T[i] + f2*t_step/2, S[i] + f2*t_step/2, c, omega, p)
        
        f4 = f(T[i] + f3*t_step, S[i] + f3*t_step, c, omega, p)
        g4 = g(T[i] + f3*t_step, S[i] + f3*t_step, c, omega, p)

        T[i + 1] = T[i] + t_step*(1/6)*(f1 + 2*f2 + 2*f3 + f4)
        S[i + 1] = S[i] + t_step*(1/6)*(g1 + 2*g2 + 2*g3 + g4)
        
        psi_[i+1] = psi(T[i], S[i], c)
        
    return t, T, S, psi_
  
#%%

[t, T_1, S_1, psi_1] = RK4_GH08(50, 10000, 15, 35, 200e-7, 0, 0)
[t, T_2, S_2, psi_2] = RK4_GH08(50, 10000, 15, 35, 300e-7, 0, 0)
[t, T_3, S_3, psi_3] = RK4_GH08(50, 10000, 15, 35, 400e-7, 0, 0)
[t, T_4, S_4, psi_4] = RK4_GH08(50, 10000, 15, 35, 550e-7, 0, 0)
[t, T_5, S_5, psi_5] = RK4_GH08(50, 10000, 15, 35, 650e-7, 0, 0)
[t, T_6, S_6, psi_6] = RK4_GH08(50, 10000, 15, 35, 800e-7, 0, 0)

plt.rcParams['figure.figsize'] = [10, 10]
plt.plot(t, T_1, linewidth = 4, color = 'black', label = "c = 200$x10^{-7}$")
plt.plot(t, T_2, linewidth = 4, color = 'blue', label = "c = 300$x10^{-7}$")
plt.plot(t, T_3, linewidth = 4, color = 'green', label = "c = 400$x10^{-7}$")
plt.plot(t, T_4, linewidth = 4, color = 'goldenrod', label = "c = 550$x10^{-7}$")#plt.xlim([0, 50])
plt.plot(t, T_5, linewidth = 4, color = 'orange', label = "c = 650$x10^{-7}$")
plt.plot(t, T_6, linewidth = 4, color = 'red', label = "c = 800$x10^{-7}$")#plt.xlim([0, 50])
plt.xlim([0, 50])
plt.ylim([6, 18])
plt.xlabel("Integration Time", fontsize = 25)
plt.ylabel('Temperature ($^{o}$C)', fontsize = 25)
plt.xticks(np.arange(0, 60, 10), fontsize = 25)
plt.yticks(np.arange(6, 20, 2), fontsize = 25)
plt.legend(fontsize = 18)
plt.text(1.5, 17, 'a)', fontsize = 40, weight='bold')
plt.show

#%%

[t, T_1, S_1, psi_1] = RK4_GH08(50, 10000, 15, 35, 400e-7, 0, 0)
[t, T_2, S_2, psi_2] = RK4_GH08(50, 10000, 15, 35, 400e-7, 1e-3, 0)
[t, T_3, S_3, psi_3] = RK4_GH08(50, 10000, 15, 35, 400e-7, 2e-3, 0)
[t, T_4, S_4, psi_4] = RK4_GH08(50, 10000, 15, 35, 400e-7, 4e-3, 0)
[t, T_5, S_5, psi_5] = RK4_GH08(50, 10000, 15, 35, 400e-7, 8e-3, 0)
[t, T_6, S_6, psi_6] = RK4_GH08(50, 10000, 15, 35, 400e-7, 16e-3, 0)

plt.rcParams['figure.figsize'] = [10, 10]
plt.axhline(0, linestyle = '--', color = 'black')
plt.plot(t, psi_1, linewidth = 4, color = 'black', label = "$\omega = 0$")
plt.plot(t, psi_2, linewidth = 4, color = 'blue', label = "$\omega = 1x10^{-3}$")
plt.plot(t, psi_3, linewidth = 4, color = 'green', label = "$\omega = 2x10^{-3}$")
plt.plot(t, psi_4, linewidth = 4, color = 'goldenrod', label = "$\omega = 4x10^{-3}$")#plt.xlim([0, 50])
plt.plot(t, psi_5, linewidth = 4, color = 'orange', label = "$\omega = 8x10^{-3}$")
plt.plot(t, psi_6, linewidth = 4, color = 'red', label = "$\omega = 16x10^{-3}$")#plt.xlim([0, 50])
plt.xlim([0, 60])
plt.ylim([-0.2, 0.2])
plt.xlabel("Integration Time", fontsize = 25)
plt.ylabel('Flow Rate ($\psi$)', fontsize = 25)
plt.xticks(np.arange(0, 70, 10), fontsize = 25)
plt.yticks(np.arange(-0.2, 0.3, 0.1), fontsize = 25)
plt.legend(loc = 'lower right', fontsize = 18)
plt.text(1.5, 0.16, 'b)', fontsize = 40, weight='bold')
plt.show
