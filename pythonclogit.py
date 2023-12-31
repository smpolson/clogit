import numpy as np
import pandas as pd

def LL(p, model, y, group):
    #compute xb for each horse
    xb = np.exp(np.sum((p*model), axis = 1, keepdims = True))
    
    # Sum xb, by group (race)
    unique_groups = np.unique(group)
    LL_p2 = []
    for item in unique_groups:
        LL_p2.append(xb[group == item].sum())
    LL_p2 = np.log(LL_p2)

    #sum xb*win for LL fxn
    xb_win = np.multiply(np.log(xb), y)
    LL_p1 = []
    for item in unique_groups:
        LL_p1.append(xb_win[group == item].sum())
        
    #compute LL    
    LL = np.subtract(LL_p1, LL_p2) 
    return np.sum(LL)

# equation 19 of McFadden, returns vector of length equal to number of weights
def dLL(p, model, y, group):    
    # Calculating P vector
    unique_groups = np.unique(group)
    P = np.empty_like(model)
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        denominator = np.sum(np.exp(model[group_indices] @ p), axis=0)
        P[group_indices] = np.exp(model[group_indices] @ p)[:, np.newaxis] / denominator
    
    # dLL calculation
    unique_groups = np.unique(group)
    dLL = []
    S = y
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        dLL.append(np.sum(((S[group_indices] - P[group_indices]) * (model[group_indices])), axis = 0))
        
    return np.sum(dLL, axis = 0)

# i index denotes the first beta the differentiation is with respect to
# equation 20 of McFadden
def ddLL(p, model, y, group, i):
    # Calculating P vector, used this method to avoid exp overflow
    unique_groups = np.unique(group)
    P = np.empty_like(model)
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        denominator = np.sum(np.exp(model[group_indices] @ p), axis=0)
        P[group_indices] = np.exp(model[group_indices] @ p)[:, np.newaxis] / denominator
    
    # Calculate ddLL
    unique_groups = np.unique(group)
    ddLL = []
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        z_avg = np.sum(model[group_indices] * P[group_indices], axis = 0)
        first = np.subtract((model[group_indices].T)[i], z_avg[i])
        first = np.tile(first, (len(p), 1)).T
        second = P[group_indices]
        ddLL.append(np.sum(first * second * (np.subtract(model[group_indices], z_avg.T)), axis = 0))

    return -np.sum(ddLL, axis = 0)

def jacobian(p, model, y, group):
    N = len(p)
    jac = np.empty((N, N))
    for i in range(N):
        ddLL_i = ddLL(p, model, y, group, i)
        jac[i] = ddLL_i
    return jac

df = pd.read_stata("E:/Horses/Analysis/2023.08.05 clogit investigation/Data/python input.dta")
print(df.columns)
# model = df[['ln_implied_prob','pole_adj_track_new','pole_adj_main','ema_past_bsn_n','combo_th','prev_glicko1','prev_glicko2','prev_glicko3','gap_bsn_surf0','gap_bsn_surf1','mean_jockey_gap_p1','mean_jockey_gap_p2','L1_weight_dif_l3r_avg1','L1_weight_dif_l3r_avg2','max_org_bsn_L60','max_org_bsn_L61']].to_numpy()
y = df[['win']].to_numpy()
group = df[['race_id']].to_numpy()
model = df[['ln_implied_prob', 'ema_past_bsn_n']].to_numpy()
# p = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],dtype=float)
p = np.array([0, 0],dtype=float)


iteration = 0
error = 100
tol = 0.0000001  # Tolerance
max_iter = 50  # Max iterations

x_0 = np.zeros(len(p),dtype=float)

N = len(x_0)
M = len(x_0)
print(np.shape(x_0))
# Iterating until either the tolerance or max iterations is met
while np.any(abs(error) > tol) and iteration < max_iter:
    fun_evaluate = np.array([dLL(x_0, model, y, group)]).reshape(M, 1)
    
    jac = jacobian(x_0, model, y, group)
    print(jac)
    
    x_new = (x_0.reshape(-1, 1) - np.linalg.inv(jac) @ fun_evaluate).flatten()

    error = x_new - x_0
    x_0 = x_new

    print("guess")
    print(x_0)
    print(iteration)
    print(error)
    print("----")
    iteration = iteration + 1

print("The solution is")

headers = ["ln_implied_prob:", "pole_adj_track_new:", "pole_adj_main:", "ema_past_bsn_n:", "combo_th:",
           "prev_glicko1:", "prev_glicko2:", "prev_glicko3:", "gap_bsn_surf0:","gap_bsn_surf1:", "mean_jockey_gap_p1:",
           "mean_jockey_gap_p2:", "L1_weight_dif_l3r_avg1:", "L1_weight_dif_l3r_avg2:", "max_org_bsn_L60:", "max_org_bsn_L61:"]

max_header_length = max(len(header) for header in headers)

for header, value in zip(headers, x_new):
    print(f"{header:<{max_header_length}} | {value}")


#EOF#
