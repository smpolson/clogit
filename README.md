# Clogit + Rologit Python Replication
[Clogit File Breakdown](#clogit-code-breakdown)\
[Rologit File Breakdown](#rologit-code-breakdown)
## Build Status
Currently working on finding optimial temperature parameter for rologit implementation. Temperature is hard coded right now.
## Clogit Code Breakdown
First, here is the log likelihood function. This is a direct replication of the function stata's clogit command uses and confirmed that this matches with stata's implementation.
```
def LL(p, model, y, group):
    xb = np.exp(np.sum((p*model), axis = 1, keepdims = True))
    unique_groups = np.unique(group)
    LL_p2 = []
    for item in unique_groups:
        LL_p2.append(xb[group == item].sum())
    LL_p2 = np.log(LL_p2)
    xb_win = np.multiply(np.log(xb), y)
    LL_p1 = []
    for item in unique_groups:
        LL_p1.append(xb_win[group == item].sum()) 
    LL = np.subtract(LL_p1, LL_p2) 
    return np.sum(LL)
```
Next, dLL calculates the first partial of the log likelihood function based on equation 19 of McFadden (1974) paper. Here, parameter index indicates which weight the partial is with respect to. This method will returns a vector of the partials. Code:
```
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
```
Relevant equation:
$$\frac{\partial L}{\partial \theta} = \sum_{n = 1}^N [\sum_{j = 1}^{J_n} (S_{jn} - P_{jn})z_{jn}]$$
Next, this is the second partial of the log likelihood function as in equation 20 of McFadden (1974) paper. This method returns a vector of partials first with respect to the ith beta (index i of vector p).
```
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
```
Based on this equation:
$$\frac{\partial^2 L}{\partial \theta \partial \theta^{\prime}} = -\sum_{n = 1}^N \sum_{j = 1}^{J_n} (z_{jn} - \overline{z_n})^{\prime}  P_{jn} (z_{jn} - \overline{z_n})$$
where 
$$\overline{z_n} = \sum_{i=1}^{J_n} z_{in}P_{in}$$
Next, the Jacobian function:
```
def jacobian(p, model, y, group):
    N = len(p)
    jac = np.empty((N, N))
    for i in range(N):
        ddLL_i = ddLL(p, model, y, group, i)
        jac[i] = ddLL_i
    return jac
```
Finally, Newton Raphson loop:
```
while np.any(abs(error) > tol) and iteration < max_iter:
    fun_evaluate = np.array([dLL(x_0, model, y, group)]).reshape(M, 1)
    jac = jacobian(x_0, model, y, group) 
    x_new = (x_0.reshape(-1, 1) - np.linalg.inv(jac) @ fun_evaluate).flatten()
    error = x_new - x_0
    x_0 = x_new
    iteration = iteration + 1
```

## Rologit Code Breakdown
The general structure of the rologit implementation is to use the clogit functions with altered inputs to calculate values for log likelihood and its partials. Note: Initially, I tried for a while to directly implement the log likelihood function from Stata's rologit documentation. I included it below just in case it is important in the future, but it is not in pythonrologity.py. The reason I abandonned it was I couldn't get the log likelihood to agree with Stata's:
```
def rLL(weights, model, ranks, group):
    # compute mu for each horse
    mu = np.sum(weights*model, axis = 1, keepdims = True)
    
    #sum xb for LL fxn
    unique_groups = np.unique(group)
    LL_p1 = []
    for item in unique_groups:
        LL_p1.append((mu[group == item]).sum())
    
    ##Sum xb, by group (race) only if rank of horse k worse than rank of horse j
    LL_p2 = []
    for item in unique_groups:
        intermediate = []
        indices = np.where(group == item)[0]
        for j in indices:
            race_sum = []
            for k in indices:
                if ranks[k] <= ranks[j]:
                    race_sum.append(np.exp(mu[k]))
            intermediate.append(np.log(np.sum(race_sum)))
        LL_p2.append(np.sum((intermediate)))
    
    #compute LL    
    LL = np.sum(np.subtract(LL_p1, LL_p2), axis = 0)
    return LL
```
Again, here is the function (equaiton (7) from Allison and Christakis (1994)), which Stata references:
$$\log(L) = \sum_{i = 1}^n \sum_{j = 1}^{J_i} \mu_{ij} - \sum_{i = 1}^n \sum_{j = 1}^{J_i} \log \left[ \sum_{k = 1}^{J_i} \delta_{ijk} \exp(\mu_{ik}) \right]$$
where $\delta_{ijk} = 1$ if the rank of horse k is worse than (or equal to) the rank of horse j in race i.\
\
Instead, right now the structure of the code is to calculate LL and the derivatives for the "race for first" and subsequently the "race for second". In order to do so, the "place" variable imported from Stata is 2 if the given horse won, 1 if the horse placed second, and 0 otherwise. Note that the code is currently compatible with a simple "win" variable equivalent to the clogit case as well as the "place" variable specified exactly as in the previous sentence. It is not difficult to make it compatible with "place" variable of any format (where the best place is given the highest value etc.), meaning to also take into account third place etc., I just have not done this yet.\
\
To calculate LL + partial derivative values for the "race for second", I simply remove horses that won from the data, and put this back in to the corresponding clogit functions. Here is this "data prep" step:
```
def data_prep(value, model, group, ranks):
    opponents2 = np.where(ranks != value, 1, 0).flatten()
    model = model[opponents2 == 1]
    group = group[opponents2 == 1]
    ranks = ranks[opponents2 == 1]
    return model, group, ranks
```
The functions rLL, rdLL, rddLL are all of the same structure. They first perform "first race" calculation and then use this "data_prep" function in order to do the "second race" calculation. Temperature is hard coded here to update the model before each calculation.
```
def rLL(weights, model, ranks, group):
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
    modelT2 = model / T2
    LL += obj(weights, modelT2, second, group)
    if np.max(ranks) == 0:
        return LL
    return LL
```
Otherwise, the structure is the same as clogit in terms of solving for minimum via NR loop. Currently, I am changing the format of the functions so that temperature is a variable that can be solved for, much like the weights in p.
