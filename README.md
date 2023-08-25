# Clogit + Rologit Python Replication
** Still editing **\
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
**editing**
