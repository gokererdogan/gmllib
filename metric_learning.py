# gmllib - metric learning
#
# This file contains the implementations of two closely related metric learning techniques.
# - learn_diagonal_metric: The method proposed in
#       Schultz, M., & Joachims, T. (2003).
#       Learning a Distance Metric from Relative Comparisons.
#       In Advances in Neural Information Processing Systems.
#   This method learns a diagonal metric matrix, i.e., weights for each dimension.
#
# - learn_low_rank_metric: This method is a variant of the above method that assumes
#       the metric matrix is low rank, i.e., A_{dxd} = G^T G where G rxd.
#
# Feb. 9, 2016
# https://github.com/gokererdogan

import numpy as np
import scipy.optimize as optim

import helpers


def _mahalonobis_distance(x, y, A, type='full'):
    """
    Calculates the distance between x and y with respect to the metric defined by A: x^T*A*y

    Parameters:
        x (numpy.ndarray)
        y (numpy.ndarray)
        A (numpy.ndarray): Distance metric matrix. Vector or 2D matrix.
        type (string): The type of distance metric matrix provided.
            One of 'full', 'low_rank' or 'diag'
            'full': d(x, y) = (x-y)^T A (x-y), A is dxd
            'low_rank': d(x, y) = (x-y)^T G^T G (x-y), A is rxd
            'diag': d(x, y) = (x-y)^T np.diag(A) (x-y) where A is dx1 

    Returns:
        numpy.ndarray: x - y
        float: Distance between x and y with respect to A
    """
    d = x - y
    if type == 'full':
        dist = np.dot(d, np.dot(A, d.T))
    elif type == 'low_rank':
        v = np.dot(A, d)
        dist = np.sum(np.square(v))
    elif type == 'diag':
        dist = np.sum(np.square(d) * A)
    else:
        raise ValueError("Unknown value for parameter type.")
    return d, dist


def _get_relative_constraints(x, A):
    """
    Construct the set of relations defined on x by metric defined by A.

    Parameters:
        x (numpy.ndarray): Data matrix
        A (numpy.ndarray): Distance metric matrix

    Returns:
        list: List of relations of the form (i, j, k)
    """
    N = x.shape[0]
    relations = []
    for i in range(N):
        for j in range(N):
            if j != i:
                for k in range(j):
                    if k != i:
                        dij, dist_ij = _mahalonobis_distance(x[i, :], x[j, :], A)
                        dik, dist_ik = _mahalonobis_distance(x[i, :], x[k, :], A)
                        if dist_ij < dist_ik:
                            relations.append((i, j, k))
                        else:
                            relations.append((i, k, j))
    return relations


def calculate_objective_function_value(x, A, relations, cost, type='full'):
    """
    Calculates the objective function value for the metric learning problem.
        0.5*tr(A^T A) + cost*\sum max(dist_ij - dist_ik + 1, 0)

    Parameters:
        x (numpy.ndarray): Data matrix
        A (numpy.ndarray): Learned distance metric matrix
        relations (list): Relative similarity constraints of the form (i, j, k)
        cost: Cost parameter
        type (string): The type of distance metric matrix provided.
            One of 'full', 'low_rank' or 'diag'
            'full': d(x, y) = (x-y)^T A (x-y), A is dxd
            'low_rank': d(x, y) = (x-y)^T G^T G (x-y), A is rxd
            'diag': d(x, y) = (x-y)^T np.diag(A) (x-y) where A is dx1 

    Returns:
        float: Objective function value
        float: Value of the first term in the objective function (depending on A)
        float: Value of the second term in the objective function (depending on relations)
    """
    if type == 'full':
        f_w = np.trace(np.dot(A.T, A)) / 2.0
    elif type == 'low_rank':
        f_w = np.trace(np.dot(A, A.T)) / 2.0
    elif type == 'diag':
        f_w = np.sum(np.square(A)) / 2.0
    else:
        raise ValueError("Unknown value for parameter type.")

    f_e = 0.0
    for i, j, k in relations:
        dij, dist_ij = _mahalonobis_distance(x[i, :], x[j, :], A, type=type)
        dik, dist_ik = _mahalonobis_distance(x[i, :], x[k, :], A, type=type)
        if dist_ij - dist_ik + 1.0 > 0.0:
            f_e += (cost * (dist_ij - dist_ik + 1.0))
    return f_w + f_e, f_w, f_e


def calculate_accuracy(x, A, relations, type='full'):
    """
    Calculates the accuracy of the learned metric A on data x and relations. Note that a relation (i, j, k) is
    predicted correctly when dist_ij < dist_ik.

    Parameters:
        x (numpy.ndarray): Data matrix
        A (numpy.ndarray): Learned distance metric matrix
        relations (list): Relative similarity constraints of the form (i, j, k)
        type (string): The type of distance metric matrix provided.
            One of 'full', 'low_rank' or 'diag'
            'full': d(x, y) = (x-y)^T A (x-y), A is dxd
            'low_rank': d(x, y) = (x-y)^T G^T G (x-y), A is rxd
            'diag': d(x, y) = (x-y)^T np.diag(A) (x-y) where A is dx1 

    Returns:
        float: Accuracy, i.e., the percentage of correctly predicted constraints.
    """
    correct = 0.0
    for (i, j, k) in relations:
        dij, dist_ij = _mahalonobis_distance(x[i, :], x[j, :], A, type=type)
        dik, dist_ik = _mahalonobis_distance(x[i, :], x[k, :], A, type=type)
        if dist_ij < dist_ik:
            correct += 1.0
    return correct / len(relations)


def _learn_diagonal_metric_scipy_optimize(method, cost, dist_mat, dist_squared, max_iter, tol, verbose):
    """
    This method is used by learn_diagonal_metric as an optimization procedure. See learn_diagonal_metric
        for details.
    """
    if method not in ['L-BFGS-B', 'TNC', 'SLSQP']:
        raise ValueError("Unknown optimization method. Should be one of L-BFGS-B, TNC, SLSQP.")

    dim_count = dist_mat.shape[0]
    relation_count = dist_mat.shape[1]

    def objective(x0, d, r):
        b = x0[0:d]
        a = x0[d:]
        obj = 0.5 * np.sum(np.square((b - np.dot(dist_mat, a)))) - np.sum(a)
        return obj

    def grad_objective(x0, d, r):
        b = x0[0:d]
        a = x0[d:]
        grad = np.zeros(d+r)
        grad[d:] = -1 - np.dot(dist_mat.T, b) + np.dot(dist_squared, a)
        grad[0:d] = -np.dot(dist_mat, a) + b
        return grad

    # initialize the optimized variables randomly
    x_initial = np.abs(np.random.randn(dim_count + relation_count)) * 0.01

    bounds_b = [(0, None)] * dim_count
    bounds_a = [(0, cost)] * relation_count
    result = optim.minimize(fun=objective, args=(dim_count, relation_count), x0=x_initial, jac=grad_objective,
                   bounds=bounds_b+bounds_a, method=method, tol=tol, options={'maxiter': max_iter, 'disp': verbose})

    alpha = result.x[dim_count:]
    beta = result.x[0:dim_count]

    return alpha, beta, result.success


def _learn_diagonal_metric_gradient_descent(cost, dist_mat, dist_squared, max_iter, tol, step, verbose):
    """
    This method is used by learn_diagonal_metric as an optimization procedure. See learn_diagonal_metric
        for details.
    """
    dim_count = dist_mat.shape[0]
    relation_count = dist_mat.shape[1]
    alpha = np.abs(np.random.randn(relation_count)) * 0.01
    beta = np.abs(np.random.randn(dim_count)) * 0.01

    converged = False
    for e in range(max_iter):
        if verbose:
            helpers.progress_bar(current=e, max=max_iter-1, update_freq=int(max_iter/100.0))

        # calculate the gradients
        grad_alpha = 1 + np.dot(dist_mat.T, beta) - np.dot(dist_squared, alpha)
        grad_beta = np.dot(dist_mat, alpha) - beta

        # check stationarity conditions
        #   if \beta_d >= 0 then grad_beta_d = 0.0
        #   if \beta_d = 0 then grad_beta_d < 0.0
        #   if cost > \alpha_r > 0 then grad_alpha_r = 0.0
        #   if \alpha_r = 0 then grad_alpha_r < 0.0
        #   if \alpha_r = cost then grad_alpha_r > 0.0
        if np.allclose(a=grad_beta[beta > 0.0], b=0.0, atol=tol) and \
                np.all(grad_beta[np.isclose(a=beta, b=0.0, atol=tol)] < 0.0) and \
                np.allclose(a=grad_alpha[np.logical_and(alpha > 0.0, alpha < cost)], b=0.0, atol=tol) and \
                np.all(grad_alpha[np.isclose(a=alpha, b=0.0, atol=tol)] < 0.0) and \
                np.all(grad_alpha[np.isclose(a=alpha, b=cost, atol=tol)] > 0.0):
            converged = True
            if verbose:
                print("\nConverged at {0:d}".format(e))
            break

        # gradient ascent update
        alpha = alpha + step * grad_alpha
        beta = beta + step * grad_beta

        # projection step
        alpha[alpha < 0.0] = 0.0
        alpha[alpha > cost] = cost
        beta[beta < 0.0] = 0.0

    return alpha, beta, converged


def learn_diagonal_metric(x, relations, cost, method, step=1e-4, max_iter=10000, tol=1e-3, verbose=False):
    """
    Learn a diagonal distance metric.
        This method solves a convex quadratic programming problem to learn a diagonal metric.
        This method was first proposed in
            Schultz, M., & Joachims, T. (2003). Learning a Distance Metric from Relative Comparisons.
            In Advances in Neural Information Processing Systems.
        We solve the following optimization problem.
            min 0.5 w^T*w + cost*\sum \ksi_{ijk}
            s.t. dist_ik - dist_ij > 1 - \ksi_{ijk}
                 w_d >= 0
                 \ksi_ijk > 0
        We form the dual of this problem
            max -0.5 (beta - D \alpha)^T (beta - D \alpha) + \sum \alpha_{ijk}
            s.t. 0 <= \alpha_{ijk} <= cost
                 0 <= \beta_d
        where \alpha and \beta are the Lagrange multipliers for relative similarity constraints and weight vector
        respectively. D is a dxR matrix where each column r contains the difference (x[i] - x[j])^2 - (x[i] - x[k])^2.
        We solve the dual using a simple projected gradient ascent procedure. The gradients are
            \frac{d}{d\alpha} = 1 - D^T D \alpha + D^t \beta
            \frac{d}{d\beta} = D \alpha - \beta

     Parameters:
        x (numpy.ndarray): Data matrix
        relations (list): Relative similarity constraints of the form (i, j, k)
        cost: Cost parameter
        method (string): One of 'GD', 'L-BFGS-B', 'TNC', and 'SLSQP'. 'GD' implements a simple gradient descent
            procedure.
            Rest of the methods are implemented in scipy.optimize.
            NOTE that 'GD' is much slower compared to scipy.optimize methods. It is included here as an educational
            example.
        step: Step size for the gradient ascent procedure
        max_iter: Maximum number of iterations
        tol: Tolerance for checking convergence.
        verbose: If True prints detailed information while running

    Returns:
        numpy.ndarray: Learned diagonal distance metric matrix
        (float, float, float): Objective value as calculated by calculate_objective_function_value
        float: Prediction accuracy as calculated by calculate_accuracy
        bool: True if the algorithm converged
    """
    dim_count = x.shape[1]
    relation_count = len(relations)
    dist_mat = np.zeros((dim_count, relation_count))

    # form the matrix D where each column contains the difference (x[i] - x[j])^2 - (x[i] - x[k])^2
    col = 0
    for (i, j, k) in relations:
        dij = x[i, :] - x[j, :]
        dik = x[i, :] - x[k, :]
        dist_mat[:, col] = np.square(dij) - np.square(dik)
        col += 1
    # D^T D
    dist_squared = np.dot(dist_mat.T, dist_mat)


    if method=='GD':
        alpha, beta, converged = _learn_diagonal_metric_gradient_descent(cost, dist_mat, dist_squared,
                                                                         max_iter=max_iter, tol=tol, step=step,
                                                                         verbose=verbose)
    else:
        alpha, beta, converged = _learn_diagonal_metric_scipy_optimize(method=method, cost=cost, dist_mat=dist_mat,
                                                                       dist_squared=dist_squared, max_iter=max_iter,
                                                                       tol=tol, verbose=verbose)

    # construct the learned distance weight matrix
    w = beta - np.dot(dist_mat, alpha)
    A = np.diag(w)
    objective_value = calculate_objective_function_value(x, w, relations, cost, type='diag')
    accuracy = calculate_accuracy(x, w, relations, type='diag')
    return A, objective_value, accuracy, converged


def _learn_low_rank_metric_scipy_optimize(x, relations, rank, S, cost, method, tol, max_iter, verbose):
    """
    This method is used by learn_low_rank_metric as an optimization subprocedure. See
        learn_low_rank_metric for more information.
    """
    if method in ['dogleg', 'trust-ncg']:
        raise ValueError('Cannot use optimization methods dogleg or trust-ncg.')

    def objective(x0, rnk, dim):
        g = x0.reshape(rnk, dim)
        obj = calculate_objective_function_value(x, g, relations, cost, type='low_rank')
        return obj[0]

    def grad_objective(x0, rnk, dim):
        g = x0.reshape(rnk, dim)
        # form the matrix \sum_{violated ijk} [(x[i]-x[j])(x[i]-x[j])^T - (x[i]-x[k])(x[i] - x[k])^T]
        gs = np.zeros((rnk, dim))
        for (i, j, k) in relations:
            dij, dist_ij = _mahalonobis_distance(x[i, :], x[j, :], g, type='low_rank')
            dik, dist_ik = _mahalonobis_distance(x[i, :], x[k, :], g, type='low_rank')
            if dist_ij - dist_ik + 1.0 >= 0.0:
                gs += np.dot(g, S[(i, j, k)])
        grad = g + (2 * cost * gs)
        return grad.ravel()

    # initialize the optimized variables randomly
    x_initial = np.random.randn((rank * dim_count)) * 0.01

    result = optim.minimize(fun=objective, args=(rank, dim_count), x0=x_initial, jac=grad_objective,
                            method=method, tol=tol, options={'maxiter': max_iter, 'disp': verbose})

    G = result.x.reshape((rank, dim_count))
    converged = result.success

    A = np.dot(G.T, G)
    objective_value = calculate_objective_function_value(x, G, relations, cost, type='low_rank')
    acc = calculate_accuracy(x, G, relations, type='low_rank')
    return G, converged


def _learn_low_rank_metric_gradient_descent(x, relations, rank, S, cost, tol, step, max_iter, verbose):
    """
    This method is used by learn_low_rank_metric as an optimization subprocedure. See
        learn_low_rank_metric for more information.
    """
    G = np.random.randn(rank, dim_count) * 0.01
    converged = False
    for e in range(max_iter):
        if verbose:
            helpers.progress_bar(current=e+1, max=max_iter, update_freq=int(max_iter/100))

        Gold = G
        # form the matrix \sum_{violated ijk} [(x[i]-x[j])(x[i]-x[j])^T - (x[i]-x[k])(x[i] - x[k])^T]
        GS = np.zeros((rank, dim_count))
        for (i, j, k) in relations:
            dij, dist_ij = _mahalonobis_distance(x[i, :], x[j, :], G, type='low_rank')
            dik, dist_ik = _mahalonobis_distance(x[i, :], x[k, :], G, type='low_rank')
            if dist_ij - dist_ik + 1.0 >= 0.0:
                GS += np.dot(G, S[(i, j, k)])

        # gradient descent step
        grad_G = G + (2 * cost * GS)
        G = G - (step * grad_G)

        # check convergence
        if np.sum(np.square(G - Gold)) < tol:
            converged = True
            if verbose:
                print('\nConverged at {0:d}'.format(e))
            break

    return G, converged


def learn_low_rank_metric(x, relations, rank, cost, method='TNC', step=1e-4, max_iter=10000, tol=1e-8, verbose=False):
    """
    Learn a (possibly) low rank distance metric.
        We decompose the distance metric matrix A as (G^T G). By constraining the rank of G, we can constrain the rank
        of the learned metric. One problem with this approach is that the distance function (d_ij*G^T G*d_ij) is no
        longer linear in the matrix of interest G; therefore the relative similarity constraints are no longer convex.
        Hence the problem is not convex either.
            min 0.5 tr(G^T G) + cost*\sum \ksi_{ijk}
            s.t. dist_ik - dist_ij > 1 - \ksi_{ijk}
                 \ksi_ijk > 0
        We can express this problem in the following unconstrained form
            min 0.5 tr(G^T G) + cost*\sum max(dist_ij - dist_ik + 1, 0)
        We solve this problem using gradient descent. The gradient with respect to G is
            G + 2*cost*\sum_{violated ijk} G [(x[i]-x[j])(x[i]-x[j])^T - (x[i]-x[k])(x[i] - x[k])^T]

     Parameters:
        x (numpy.ndarray): Data matrix
        relations (list): Relative similarity constraints of the form (i, j, k)
        rank (int): Desired rank of the distance metric matrix
        cost: Cost parameter
        method: 'GD', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP'. 
            'GD' is an implementation of a simple gradient descent procedure included here mainly for educational
            reasons. It is much slower than other methods.
            Other methods are implemented through scipy.optimize package.
        step: Step size for the gradient ascent procedure
        max_iter: Maximum number of iterations
        tol: Tolerance for checking convergence.
        verbose: If True prints detailed information while running

    Returns:
        numpy.ndarray: Learned diagonal distance metric matrix
        (float, float, float): Objective value as calculated by calculate_objective_function_value
        float: Prediction accuracy as calculated by calculate_accuracy
        bool: True if the algorithm converged
    """
    dim_count = x.shape[1]
    S_ijk = {}
    for (i, j, k) in relations:
        dij, _ = _mahalonobis_distance(x[i, :], x[j, :], np.eye(dim_count))
        dik, _ = _mahalonobis_distance(x[i, :], x[k, :], np.eye(dim_count))
        S_ijk[(i, j, k)] = np.outer(dij, dij) - np.outer(dik, dik)

    if method=='GD':
        G, converged = _learn_low_rank_metric_gradient_descent(x, relations, rank, S_ijk, cost=cost, tol=tol, step=step,
                                                               max_iter=max_iter, verbose=verbose)
    else:
        G, converged = _learn_low_rank_metric_scipy_optimize(x, relations, rank, S_ijk, cost=cost, method=method,
                                                             tol=tol, max_iter=max_iter, verbose=verbose)

    A = np.dot(G.T, G)
    objective_value = calculate_objective_function_value(x, G, relations, cost, type='low_rank')
    acc = calculate_accuracy(x, G, relations, type='low_rank')
    return A, objective_value, acc, converged


if __name__ == '__main__':
    N = 10
    dim_count = 10
    rank = 2
    x = np.random.randn(N, dim_count)
    G = np.random.randn(rank, dim_count)
    A = np.dot(G.T, G)
    relations = _get_relative_constraints(x, A)

    # TEST PROBLEMS
    # # A = 1.0 <- solution
    # x = np.array([0.0, 0.0, 1.0, np.sqrt(0.5)])
    # x.shape = (4, 1)
    # relations = [(0, 1, 2), (0, 1, 3)]

    # x = np.array([0.0, 0.0, 1.0])
    # x.shape = (3, 1)
    # relations = [(0, 1, 2)]

    np.random.shuffle(relations)
    N_train = int(len(relations) / 2.0)

    cost = 1.0
    step = 0.0001

    diag_A, diag_obj, diag_acc, diag_converged = learn_diagonal_metric(x, relations[0:N_train], cost=cost, method='GD',
                                                                       step=step, tol=1e-3, max_iter=100000,
                                                                       verbose=True)
    diag_test_acc = calculate_accuracy(x, diag_A, relations[N_train:])

    diag_A2, diag_obj2, diag_acc2, diag_converged2 = learn_diagonal_metric(x, relations[0:N_train], cost=cost, 
                                                                           method='SLSQP', tol=1e-6, max_iter=10000, 
                                                                           verbose=True)
    diag_test_acc2 = calculate_accuracy(x, diag_A2, relations[N_train:])

    lr_A, lr_obj, lr_acc, lr_converged = learn_low_rank_metric(x, relations[0:N_train], rank=2, cost=cost, method='TNC',
                                                               tol=1e-6, max_iter=20000, verbose=True)
    lr_test_acc = calculate_accuracy(x, lr_A, relations[N_train:])

    lr_A2, lr_obj2, lr_acc2, lr_converged2 = learn_low_rank_metric(x, relations[0:N_train], rank=2, cost=cost,
                                                                   method='GD', step=step, tol=1e-9, max_iter=10000,
                                                                   verbose=True)
    lr_test_acc2 = calculate_accuracy(x, lr_A2, relations[N_train:])

    print('Diagonal with GD:', diag_obj, diag_acc, diag_test_acc, diag_converged)
    print('Diagonal with SLSQP: ', diag_obj2, diag_acc2, diag_test_acc2, diag_converged2)
    print('Low-rank with GD: ', lr_obj2, lr_acc2, lr_test_acc2, lr_converged2)
    print('Low-rank with TNC: ', lr_obj, lr_acc, lr_test_acc, lr_converged)

    print('Objective value with true metric: ', calculate_objective_function_value(x, A, relations[0:N_train], cost))
    print('Accuracy wit Euclidean distance: ', calculate_accuracy(x, np.eye(dim_count), relations[N_train:]))
