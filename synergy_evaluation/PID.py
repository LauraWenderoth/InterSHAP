from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import cvxpy as cp
from cvxpy import *
import numpy as np
from scipy.special import rel_entr
import pickle
def solve_Q_new(P: np.ndarray):
    '''
    Compute optimal Q given 3d array P
    with dimensions coressponding to x1, x2, and y respectively
    '''
    Py = P.sum(axis=0).sum(axis=0)
    Px1 = P.sum(axis=1).sum(axis=1)
    Px2 = P.sum(axis=0).sum(axis=1)
    Px2y = P.sum(axis=0)
    Px1y = P.sum(axis=1)
    Px1y_given_x2 = P / P.sum(axis=(0, 2), keepdims=True)

    Q = [cp.Variable((P.shape[0], P.shape[1]), nonneg=True) for i in range(P.shape[2])]
    Q_x1x2 = [cp.Variable((P.shape[0], P.shape[1]), nonneg=True) for i in range(P.shape[2])]

    # Constraints that conditional distributions sum to 1
    sum_to_one_Q = cp.sum([cp.sum(q) for q in Q]) == 1

    # Brute force constraints #
    # [A]: p(x1, y) == q(x1, y)
    # [B]: p(x2, y) == q(x2, y)

    # Adding [A] constraints
    A_cstrs = []
    for x1 in range(P.shape[0]):
        for y in range(P.shape[2]):
            vars = []
            for x2 in range(P.shape[1]):
                vars.append(Q[y][x1, x2])
            A_cstrs.append(cp.sum(vars) == Px1y[x1, y])

    # Adding [B] constraints
    B_cstrs = []
    for x2 in range(P.shape[1]):
        for y in range(P.shape[2]):
            vars = []
            for x1 in range(P.shape[0]):
                vars.append(Q[y][x1, x2])
            B_cstrs.append(cp.sum(vars) == Px2y[x2, y])

    # KL divergence
    Q_pdt_dist_cstrs = [cp.sum(Q) / P.shape[2] == Q_x1x2[i] for i in range(P.shape[2])]

    # objective
    obj = cp.sum([cp.sum(cp.rel_entr(Q[i], Q_x1x2[i])) for i in range(P.shape[2])])
    # print(obj.shape)
    all_constrs = [sum_to_one_Q] + A_cstrs + B_cstrs + Q_pdt_dist_cstrs
    prob = cp.Problem(cp.Minimize(obj), all_constrs)
    try:
        prob.solve(verbose=False, max_iters=10000,solver=cp.ECOS)
    except:
        prob.solve(solver=cp.SCS, verbose=True, max_iters=10000)

    # print(prob.status)
    # print(prob.value)
    # for j in range(P.shape[1]):
    #  print(Q[j].value)

    return np.stack([q.value for q in Q], axis=2)
def MI(P: np.ndarray):
  ''' P has 2 dimensions '''
  margin_1 = P.sum(axis=1)
  margin_2 = P.sum(axis=0)
  outer = np.outer(margin_1, margin_2)

  return np.sum(rel_entr(P, outer))
  # return np.sum(P * np.log(P/outer))

def CoI(P:np.ndarray):
  ''' P has 3 dimensions, in order X1, X2, Y '''
  # MI(Y; X1)
  A = P.sum(axis=1)

  # MI(Y; X2)
  B = P.sum(axis=0)

  # MI(Y; (X1, X2))
  C = P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1]))

  return MI(A) + MI(B) - MI(C)

def CI(P, Q):
  assert P.shape == Q.shape
  P_ = P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1]))
  Q_ = Q.transpose([2, 0, 1]).reshape((Q.shape[2], Q.shape[0]*Q.shape[1]))
  return MI(P_) - MI(Q_)

def UI(P, cond_id=0):
  ''' P has 3 dimensions, in order X1, X2, Y
  We condition on X1 if cond_id = 0, if 1, then X2.
  '''
  P_ = np.copy(P)
  sum = 0.

  if cond_id == 0:
    J= P.sum(axis=(1,2)) # marginal of x1
    for i in range(P.shape[0]):
      sum += MI(P[i,:,:]/P[i,:,:].sum()) * J[i]
  elif cond_id == 1:
    J= P.sum(axis=(0,2)) # marginal of x1
    for i in range(P.shape[1]):
      sum += MI(P[:,i,:]/P[:,i,:].sum()) * J[i]
  else:
    assert False

  return sum
def get_measure(P,do_round=True,metric='PID'):
  Q = solve_Q_new(P)
  redundancy = CoI(Q)
  unique_1 = UI(Q, cond_id=1)
  unique_2 = UI(Q, cond_id=0)
  synergy = CI(P, Q)

  if do_round:
      redundancy = round(redundancy, 2)
      synergy = round(synergy, 2)
      unique_2 = round(unique_2, 2)
      unique_1 = round(unique_1, 2)
  print('Redundancy', redundancy)
  print('Unique 0', unique_1)
  print('Unique 1', unique_2)
  print('Synergy', synergy)

  return {f'{metric}_redundancy':redundancy, f'{metric}_unique0':unique_1, f'{metric}_unique1':unique_2, f'{metric}_synergy':synergy}

def extract_categorical_from_data(x):
  supp = set(x)
  raw_to_discrete = dict()
  for i in supp:
    raw_to_discrete[i] = len(raw_to_discrete)
  discrete_data = [raw_to_discrete[x_] for x_ in x]

  return discrete_data, raw_to_discrete
def convert_data_to_distribution(x1: np.ndarray, x2: np.ndarray, y: np.ndarray):
    assert x1.size == x2.size
    assert x1.size == y.size

    numel = x1.size

    x1_discrete, x1_raw_to_discrete = extract_categorical_from_data(x1.squeeze())
    x2_discrete, x2_raw_to_discrete = extract_categorical_from_data(x2.squeeze())
    y_discrete, y_raw_to_discrete = extract_categorical_from_data(y.squeeze())

    joint_distribution = np.zeros((len(x1_raw_to_discrete), len(x2_raw_to_discrete), len(y_raw_to_discrete)))
    for i in range(numel):
        joint_distribution[x1_discrete[i], x2_discrete[i], y_discrete[i]] += 1
    joint_distribution /= np.sum(joint_distribution)

    return joint_distribution, (x1_raw_to_discrete, x2_raw_to_discrete, y_raw_to_discrete)

def clustering(X, pca=False, n_clusters=20, n_components=5):
    X = np.nan_to_num(X)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0],-1)
    if pca:
        # print(np.any(np.isnan(X)), np.all(np.isfinite(X)))
        X = normalize(X)
        X = PCA(n_components=n_components).fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters,n_init=10).fit(X)
    return kmeans.labels_, X

if __name__ == "__main__":
    data_path = ''
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

