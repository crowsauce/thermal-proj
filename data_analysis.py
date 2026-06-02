import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd

def analyse_data(T, n): # turns out the data was in r/t not r^2/t! its chill tho i can fix it. shit took way to long i am NOT running the sims again
    data = pd.read_csv(f"data/{T}K,{n}.csv", header=0, names=["r/t"])
    data["r2/t"] = data["r/t"]**2*data.index**10**(-7)
    data_cut = data.iloc[-500:,:]
    mean_data = data_cut["r2/t"].mean()
    std_data = data_cut["r2/t"].std()
    return mean_data, std_data

T_range = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3500, 4000]

def run_data(T_range):
    row_list = []
    for T in T_range:   
        for n in [1,2,3,4,5,6,7,8,9,10]:
            mean_data, std_data = analyse_data(T, n)
            row_list.append({"T": T, "mean r2/t": mean_data, "std r2/t": std_data})
    data = pd.DataFrame(row_list)
    return data

data_set_fr = run_data(T_range)

def predicted_rel(T, S, a):
    r2 = a*T**(1/2) + a*S/T**(1/2)
    return r2

def lin_fit(x, grad, intercept):
    y = grad*x + intercept
    return y

parameters, covariance = scipy.optimize.curve_fit(predicted_rel, data_set_fr["T"], data_set_fr["mean r2/t"])
S_fit = parameters[0]
a = parameters[1]
S_error = np.sqrt(covariance[0][0])
a_error = np.sqrt(covariance[1][1])
print(f"Fitted S value: {S_fit}")
print(f"Fitted a value: {a}")
print(f"Standard error for S: {S_error}")
print(f"Standard error for a: {a_error}")
fittedT = np.linspace(min(data_set_fr["T"]), max(data_set_fr["T"]), 100)
fitted_r2 = predicted_rel(fittedT, S_fit, a)
low_fit_r2 = predicted_rel(fittedT, S_fit - S_error, a - a_error)
high_fit_r2 = predicted_rel(fittedT, S_fit + S_error, a + a_error)

#linslope, linintercept, rvalue, pvalue, stderr = scipy.stats.linregress(data_set_fr["T"], data_set_fr["mean r2/t"])
#print(linslope, linintercept, rvalue, pvalue, stderr)
#lin_fit_T = np.linspace(min(data_set_fr["T"]), max(data_set_fr["T"]), 100)
#lin_fit_r2 = lin_fit(lin_fit_T, linslope, linintercept)

plt.errorbar(data_set_fr["T"], data_set_fr["mean r2/t"], yerr=data_set_fr["std r2/t"], fmt='o')
plt.plot(fittedT, fitted_r2, color='orange', label='S=45.9, a=2.0e-8')
plt.plot(fittedT, low_fit_r2, color='yellow', label='S=-132.9, a=1.7e-8')
plt.plot(fittedT, high_fit_r2, color='red', label='S=224.8, a=2.3e-8')

plt.xlabel("Temperature (K)")
plt.ylabel("Mean r^2/t (m^2/s)")
plt.title("Mean r^2/t vs Temperature")
plt.legend()
plt.show()