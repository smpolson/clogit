# Clogit Python Replication
** Still editing **
## Build Status
One iteration of the NR loop successfully runs before on the second loop the Jacobian matrix is all zeros, which breaks the loop. This most likely is the result of an error within dLL and/or ddLL implementations.
## Code Breakdown
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
Next, dLL calculates the first partial of the log likelihood function based on equation 19 of McFadden (1974) paper. Here, parameter index indicates which weight the partial is with respect to -- to make this implementation more efficient, I will alter this so that instead of calling dLL by each index, this method will return a vector instead. Note that the P vector is calculated this way in order to avoid exp overflow. Code:
```
def dLL(p, model, y, group, index):
    max_utility = np.max(p @ model.T)
    scaled_utilities = np.exp((p @ model.T) - max_utility)
    denominator = np.sum(scaled_utilities)
    log_denominator = np.log(denominator) + max_utility
    log_probabilities = (p @ model.T) - log_denominator
    P = np.exp(log_probabilities)

    unique_groups = np.unique(group)
    dLL = []
    for item in unique_groups:
        S = y.T * model.T[index]
        group_indices = np.where(group == item)[0]
        dLL.append(((S.T[group_indices]-P[group_indices]) @ model.T[index][group_indices]).sum())
    return np.sum(dLL)
```
Relevant equation:
$\frac{\partial L}{\partial \theta} = \sum_{n = 1}^N [\sum_{j = 1}^{J_n} (S_{jn} - P_{jn})z_{jn}]$
Next, this is the second partial of the log likelihood function as in equation 20 of McFadden (1974) paper.
```
def ddLL(p, model, y, group, i, j):
    max_utility = np.max(p @ model.T)
    scaled_utilities = np.exp((p @ model.T) - max_utility)
    denominator = np.sum(scaled_utilities)
    log_denominator = np.log(denominator) + max_utility
    log_probabilities = (p @ model.T) - log_denominator
    P = np.exp(log_probabilities)
    
    unique_groups = np.unique(group)
    ddLL = []
    for item in unique_groups:
        group_indices = np.where(group == item)[0]
        z_avg_i = (model.T[i][group_indices] * P[group_indices]).sum()
        z_avg_j = (model.T[j][group_indices] * P[group_indices]).sum()
        ddLL.append((np.subtract(model.T[j][group_indices], z_avg_j)) * P[group_indices] @ (np.subtract(model.T[i][group_indices], z_avg_i)))
    return -np.sum(ddLL)
```
Based on this equation:
$\frac{\partial^2 L}{\partial \theta \partial \theta^{\prime}} = -\sum_{n = 1}^N \sum_{j = 1}^{J_n} (z_{jn} - \bar{z}_{n})^{\prime} P_{jn}(z_{jn} - \bar{z}_{n})$
where 
$\bar{z}_{n} = \sum_{i=1}^{J_n} z_{in}P{in}$
Next, Jacobian function which no doubt can be written more compactly:
```
def jacobian(p, model, y, group):
    N = len(p)
    jac = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            ddLL_i = ddLL(p, model, y, group, i, j)
            jac[i][j] = ddLL_i
    return jac
```
Finally, Newton Raphson loop. Note that NR algorithm finds zeros (not maxima / minima) so this is why both dLL and ddLL are needed. 
```
while np.any(abs(error) > tol) and iteration < max_iter:
    fun_evaluate = np.array([dLL(x_0, model, y, group, 0), dLL(x_0, model, y, group, 1)]).reshape(M, 1)
    jac = jacobian(x_0, model, y, group)
    x_new = (x_0.reshape(-1, 1) - np.linalg.inv(jac) @ fun_evaluate).flatten()
    error = x_new - x_0
    x_0 = x_new
    iteration = iteration + 1
```
