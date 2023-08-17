import numpy as np
import pandas as pd

def LL(p, model, y, group):
    #compute xb for each horse
    xb = np.exp(np.sum(p*model, axis = 1, keepdims = True))
    
    ##Sum xb, by group (race)
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
    # LL = -1*np.sum(LL) # no longer using minimize so do not need use -LL
    # print(np.sum(LL))
    return np.sum(LL)

# index denotes which beta differentiation is with respect to
# equation 19 of McFadden
def dLL(p, model, y, group, index):
    # Calculating P vector, used this method to avoid exp overflow
    max_utility = np.max(p @ model.T)
    scaled_utilities = np.exp((p @ model.T) - max_utility)
    denominator = np.sum(scaled_utilities)
    log_denominator = np.log(denominator) + max_utility
    log_probabilities = (p @ model.T) - log_denominator
    P = np.exp(log_probabilities)
    
    ##Sum xb, by group (race)
    unique_groups = np.unique(group)
    dLL = []
    for item in unique_groups:
        S = y.T * model.T[index]
        group_indices = np.where(group == item)[0]
        dLL.append(((S.T[group_indices]-P[group_indices]) @ model.T[index][group_indices]).sum())
        
    # print(np.shape(dLL))
    return np.sum(dLL)

# i and j indices denote which two betas the differentiation is with respect to
# equation 20 of McFadden
def ddLL(p, model, y, group, i, j):
    # Calculating P vector, used this method to avoid exp overflow
    max_utility = np.max(p @ model.T)
    scaled_utilities = np.exp((p @ model.T) - max_utility)
    denominator = np.sum(scaled_utilities)
    log_denominator = np.log(denominator) + max_utility
    log_probabilities = (p @ model.T) - log_denominator
    P = np.exp(log_probabilities)
    
    # Calculate ddLL
    unique_groups = np.unique(group)
    ddLL = []
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        z_avg_i = (model.T[i][group_indices] * P[group_indices]).sum()
        z_avg_j = (model.T[j][group_indices] * P[group_indices]).sum()
        ddLL.append((np.subtract(model.T[j][group_indices], z_avg_j)) * P[group_indices] @ (np.subtract(model.T[i][group_indices], z_avg_i)))
    
    return -np.sum(ddLL)

def jacobian(p, model, y, group):
    N = len(p)
    jac = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            ddLL_i = ddLL(p, model, y, group, i, j)
            jac[i][j] = ddLL_i
    return jac

df = pd.read_stata("E:/Horses/Analysis/2023.08.05 clogit investigation/Data/python input.dta")
print(df.columns)
# model = df[['ln_implied_prob','pole_adj_track_new','pole_adj_main','ema_past_bsn_n','combo_th','prev_glicko1','prev_glicko2','prev_glicko3','gap_bsn_surf0','gap_bsn_surf1','mean_jockey_gap_p1','mean_jockey_gap_p2','L1_weight_dif_l3r_avg1','L1_weight_dif_l3r_avg2','max_org_bsn_L60','max_org_bsn_L61']].to_numpy()
y = df[['win']].to_numpy()
group = df[['race_id']].to_numpy()
model = df[['ln_implied_prob', 'ema_past_bsn_n']].to_numpy()

iteration = 0
error = 100
tol = 0.00000001  # Tolerance
max_iter = 50  # Max iterations

N = 2
M = 2
x_0 = np.array([0, 0],dtype=float)
print(np.shape(x_0))
# Iterating until either the tolerance or max iterations is met
while np.any(abs(error) > tol) and iteration < max_iter:
    fun_evaluate = np.array([dLL(x_0, model, y, group, 0), dLL(x_0, model, y, group, 1)]).reshape(M, 1)
    
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
print(x_new)


#EOF#
