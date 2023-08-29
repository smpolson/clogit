import numpy as np
import pandas as pd

T1 = 1.0
T2 = 1.0

def obj(p, model, y, group):
    #compute xb for each horse
    xb = np.exp(np.sum(p*model, axis = 1, keepdims = True))
    
    ##Sum xb, by group (race)
    unique_groups = np.unique(group)
    LL_p2 = []
    for item in unique_groups:
        LL_p2.append(xb[group == item].sum())
    LL_p2 = np.log(LL_p2)

    #sum xb*win for LL fxn
    xb_win = np.log(xb) * y
    LL_p1 = []
    for item in unique_groups:
        LL_p1.append(np.sum(xb_win[group == item]))
        
    #compute LL    
    LL = np.subtract(LL_p1, LL_p2) #using minimize so need to do -LL
    LL = np.sum(LL)
    return LL


def data_prep(value, model, group, ranks):
    opponents2 = np.where(ranks != value, 1, 0).flatten()
    model = model[opponents2 == 1]
    group = group[opponents2 == 1]
    ranks = ranks[opponents2 == 1]
    return model, group, ranks

def rLL(weights, model, ranks, group):
    temperature = weights[0]
    weights = weights[1:]
    
    # note, should make a loop later, but currently should work for first and second
    num_iter = np.max(ranks)
    assert(num_iter == 2)

    LL = 0
    first = np.where(ranks == num_iter, 1, 0)


    modelT1 = model / T1
    LL += obj(weights, modelT1, first, group)

    model, group, ranks = data_prep(num_iter, model, group, ranks)

    if np.max(ranks) == 0:
        return LL
    
    second = np.where(ranks == num_iter - 1, 1, 0)
    modelT2 = model * temperature
    LL += obj(weights, modelT2, second, group)
    
    if np.max(ranks) == 0:
        return LL

    return LL

# equation 19 of McFadden, returns vector of length equal to number of weights
def dLL(p, model, y, group):    
    # Calculating P vector
    unique_groups = np.unique(group)
    P = np.empty_like(model)
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        denominator = np.sum(np.exp(model[group_indices] @ p), axis=0)
        P[group_indices] = np.exp(model[group_indices] @ p)[:, np.newaxis] / denominator
    
    ##Sum xb, by group (race)
    unique_groups = np.unique(group)
    dLL = []
    S = y
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        dLL.append(np.sum(((S[group_indices] - P[group_indices]) * (model[group_indices])), axis = 0))
        
    return np.sum(dLL, axis = 0)


def dT(temperature, p, model, y, group):
    beta = 1 * temperature
    
    # Calculating P vector
    unique_groups = np.unique(group)
    P = np.empty_like(model)
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        denominator = np.sum(np.exp(beta * (model[group_indices] @ p)), axis=0)
        P[group_indices] = np.exp(beta * (model[group_indices] @ p))[:, np.newaxis] / denominator
    
    ##Sum xb, by group (race)
    unique_groups = np.unique(group)
    dLL = []
    S = y
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        dLL.append(np.sum(((S[group_indices] - P[group_indices]) * (model[group_indices])), axis = 0))
        
    # so here I choose to sum all of them to fix the dimension issue, but this could be wrong
    return np.sum(np.sum(dLL, axis = 0), axis = 0)


def rdLL(weights, model, ranks, group):
    temperature = weights[0]
    dLL2 = np.zeros(len(weights))
    weights = weights[1:]
    
    # note, should make a loop later, but currently should work for first and second
    num_iter = np.max(ranks)
    
    first = np.where(ranks == num_iter, 1, 0)
    modelT1 = model / T1
    dLL2[1:] += dLL(weights, modelT1, first, group)
    
    # now prep for second
    model, group, ranks = data_prep(num_iter, model, group, ranks)

    if np.max(ranks) == 0:
        return dLL2
    
    # to calculate
    dLL2[0] += dT(temperature, weights, model, ranks, group)
    
    second = np.where(ranks == num_iter - 1, 1, 0)
    modelT2 = model * temperature
    dLL2[1:] += dLL(weights, modelT2, second, group)
    
    if np.max(ranks) == 0:
        return dLL2
    
    return dLL2

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


def ddT(temperature, p, model, y, group, i):
    beta = 1 * temperature

    # Calculating P vector, used this method to avoid exp overflow
    unique_groups = np.unique(group)
    P = np.empty_like(model)
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        denominator = np.sum(np.exp(beta *(model[group_indices] @ p)), axis=0)
        P[group_indices] = np.exp(beta * (model[group_indices] @ p))[:, np.newaxis] / denominator
    
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
        
    return -np.sum(np.sum(ddLL, axis = 0), axis = 0)
    
    
def ddTrow(temperature, p, model, y, group, i):
    beta = 1 * temperature
    ddT = np.zeros(len(p) + 1)
    
    unique_groups = np.unique(group)
    P = np.empty_like(model)
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        denominator = np.sum(np.exp(beta *(model[group_indices] @ p)), axis=0)
        P[group_indices] = np.exp(beta * (model[group_indices] @ p))[:, np.newaxis] / denominator

    # Calculate ddLL
    unique_groups = np.unique(group)
    ddLL = []
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        z_avg = np.sum(model[group_indices] * P[group_indices], axis = 0)
        first = np.subtract((np.sum(model[group_indices].T, axis = 1)), z_avg)
        first = np.tile(first, (len(p), 1)).T
        second = P[group_indices]
        ddLL.append(np.sum(first * np.sum(second, axis = 0).T * (np.subtract((np.sum(model[group_indices].T, axis = 1)), z_avg.T)), axis = 0))
        
    ddT[0] += -np.sum(np.sum(ddLL, axis = 0), axis = 0)
    ddT[1:] += -np.sum(ddLL, axis = 0)
    return ddT


def rddLL(weights, model, ranks, group, i):
    temperature = weights[0]
    ddLL2 = np.zeros(len(weights))
    weights = weights[1:]
    
    # note, should make a loop later, but currently should work for first and second
    num_iter = np.max(ranks)
    i = i - 1
    if i >= 0:
        first = np.where(ranks == num_iter, 1, 0)
        modelT1 = model / T1
        ddLL2[1:] += ddLL(weights, modelT1, first, group, i)
    
    # now prep for second
    model, group, ranks = data_prep(num_iter, model, group, ranks)

    if np.max(ranks) == 0:
        return ddLL2
    i = i + 1

    second = np.where(ranks == num_iter - 1, 1, 0)
    if i == 0:
        ddLL2 += ddTrow(temperature, weights, model, second, group, i)
        return ddLL2
    i -= 1
    ddLL2[0] += ddT(temperature, weights, model, second, group, i)
    modelT2 = model * temperature
    ddLL2[1:] += ddLL(weights, modelT2, second, group, i)
    
    if np.max(ranks) == 0:
        return ddLL2

    return ddLL2

def rjacobian(p, model, place, group):
    N = len(p)
    jac = np.empty((N, N))
    for i in range(N):
        print(i)
        rddLL_i = rddLL(p, model, place, group, i)
        jac[i] = rddLL_i
    return jac



df = pd.read_stata("E:/Horses/Analysis/2023.08.05 clogit investigation/Data/python input.dta")
print(df.columns)
model = df[['ln_implied_prob','pole_adj_track_new','pole_adj_main','ema_past_bsn_n','combo_th','prev_glicko1','prev_glicko2','prev_glicko3','gap_bsn_surf0','gap_bsn_surf1','mean_jockey_gap_p1','mean_jockey_gap_p2','L1_weight_dif_l3r_avg1','L1_weight_dif_l3r_avg2','max_org_bsn_L60','max_org_bsn_L61']].to_numpy()
y = df[['win']].to_numpy()
group = df[['race_id']].to_numpy()
#model = df[['ln_implied_prob', 'ema_past_bsn_n']].to_numpy()
place = df[['place']].to_numpy()

# first index denotes temperature
p = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],dtype=float)

iteration = 0
error = 100
tol = 0.0000001  # Tolerance
max_iter = 50  # Max iterations


#x_0 = np.zeros(len(p),dtype=float)
x_0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],dtype=float)

N = len(x_0)
M = len(x_0)

# Iterating until either the tolerance or max iterations is met
while np.any(abs(error) > tol) and iteration < max_iter:
    fun_evaluate = np.array([rdLL(x_0, model, place, group)]).reshape(M, 1)
    
    jac = rjacobian(x_0, model, place, group)
    print(jac)
    
    x_new = (x_0.reshape(-1, 1) - np.linalg.inv(jac) @ fun_evaluate).flatten()

    error = x_new - x_0
    x_0 = x_new

    print("guess")
    print(x_0)
    print(iteration)
    print(error)
    print("----")
    print(rLL(x_0, model, place, group))
    iteration = iteration + 1

print("The solution is")

headers = ["Temperature", "ln_implied_prob:", "pole_adj_track_new:", "pole_adj_main:", "ema_past_bsn_n:", "combo_th:",
           "prev_glicko1:", "prev_glicko2:", "prev_glicko3:", "gap_bsn_surf0:","gap_bsn_surf1:", "mean_jockey_gap_p1:",
           "mean_jockey_gap_p2:", "L1_weight_dif_l3r_avg1:", "L1_weight_dif_l3r_avg2:", "max_org_bsn_L60:", "max_org_bsn_L61:"]

max_header_length = max(len(header) for header in headers)

for header, value in zip(headers, x_new):
    print(f"{header:<{max_header_length}} | {value}")
