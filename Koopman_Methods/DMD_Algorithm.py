# General DMD Algorithm using PCA and Delay Embeddings

import numpy as np
import time
import matplotlib.pyplot as plt

# time-series used in data collection
dt = 0.067

# testing predictive power of DMD model
start = 20
steps = 70
t_s = np.linspace(0,steps*dt,steps)

# load the data (assumes format as each variable assigned to a column)
data = (np.transpose(np.genfromtxt('<insert data path>', skip_header=True, delimiter=',', dtype='float')))[:,start:]

def PCA(M):
# Centering the data
    M_centered = M - np.mean(M, axis=0)
    C_X = np.cov(M_centered)
    eigenvalues, eigenvectors = np.linalg.eig(C_X)
    ncols = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:,ncols]
    return W

# transforming to the optimal basis where Covariance matrix is diagonal
W = PCA(data)
data = np.dot(W.T,data)

# Hankel DMD for spatially undersampled data (delay embeddings), s<=n
s = 50
m, n = data.shape
print(f'\nDimensions of Data matrix: {m}x{n}')

arr = []
for i in range(n-s+1):
    col = data[:,i]
    for j in range(s-1):
        cols = np.hstack((col,data[:,j+i+1]))
        col = cols
    row = cols.reshape(-1,1)
    arr.append(row)
data_aug = np.array(np.hstack(arr))

X_0 = data_aug[:,:-1]
X_1 = data_aug[:,1:]

m_X0, n_X0 = X_0.shape
print(f'\nDimensions of X_0: {m_X0}x{n_X0}')

# SVD

U, S_s, Vt = np.linalg.svd(X_0, full_matrices=False)
S = np.diag(S_s)

m_U, n_U = U.shape
print(f'\nDimensions of U (X_0) matrix: {m_U}x{n_U}')
m_S, n_S = S.shape
print(f'\nDimensions of S (X_0) matrix: {m_S}x{n_S}')
m_Vt, n_Vt = Vt.shape
print(f'\nDimensions of V (X_0) matrix: {m_Vt}x{n_Vt}')

# Further truncation is useful if there are many state variables (m is large)
# A_trunc is projection of A = X_1*X_0^-1 into an r x r dim subspace

A = np.dot(X_1,np.dot(Vt.T,np.dot(np.linalg.inv(S),U.T)))
A_evals, A_evectors = np.linalg.eig(A)
ncols = np.argsort(A_evals)[::-1]
A_evals = sorted(A_evals, reverse=True)
Phi = A_evectors[:,ncols]
Lambda = np.diag(A_evals)

m_A, n_A = A.shape
print(f'\nDimensions of A: {m_A}x{n_A}')

# print(f'\nMatrix of DMD modes:\n{np.round(Phi,3)}')
# print(f'\nMatrix of DMD mode eigenvalues: \n{np.round(Lambda,3)}')

# testing predictive power of DMD model
b = np.dot(np.linalg.inv(Phi), np.array(data_aug[:,0]).reshape(-1,1))
# relationship between continuous time operator and discrete time shift operator applied to eigenvalues
w_s = np.array(np.log(np.diag(Lambda))/dt)

def eig_cont(t):
    rows, cols = Phi.shape
    x = np.zeros((1,rows), dtype='complex128')
    for j in range(cols):
        x += np.array(Phi[:,j]) * b[j] * np.exp(w_s[j]*t)
# select only first 5 entries of array - reduce down to 5-dim state space
    return np.round(x[:,:m],3)

def eig_disc(k):
    x = np.dot(Phi,np.dot(np.linalg.matrix_power(Lambda,k),b)).reshape(1,-1)
# select only first 5 entries of array - reduce down to 5-dim state space
    return np.round(x[:,:m],3)

discrete_prediction = np.empty((0,m))
continuous_prediction = np.empty((0,m))
actual = (data.T)[0:steps,:]

for index, value in enumerate(t_s):
    discrete_prediction = np.vstack((discrete_prediction, eig_disc(index)))
    continuous_prediction = np.vstack((continuous_prediction, eig_cont(value)))

print(np.round(b[:m, 0], 3))
print(discrete_prediction[0,:])
print(continuous_prediction[0,:])
print(actual[0,:])

# plotting the time evolution & absolute error evolution
for i in range(m):
    index = [x + start for x in range(len(t_s))]
    plt.plot(index, discrete_prediction[:,i], '-r', label=f'Discrete Prediction Variable {i+1}')
    plt.plot(index, continuous_prediction[:,i], '-b', label=f'Continuous Prediction Variable {i+1}')
    plt.plot(index, actual[:,i], '-k', label=f'Actual Variable {i+1}')
    disc_diff = discrete_prediction[:,i] - actual[:,i]
    cont_diff = continuous_prediction[:,i] - actual[:,i]
    plt.plot(index, disc_diff, '--r', label=f'Discrete Error Variable {i+1}')
    plt.plot(index, cont_diff, '--b', label=f'Continuous Error Variable {i+1}')
    plt.xlabel(f'Time Steps ({dt}s)')
    plt.ylabel('State Variable')
    plt.title(f'State Variable {i+1} DMD Evolution, Delay Embedding s={s}')
    plt.legend(loc='upper right')
    plt.grid()
    # plt.savefig(f'variable_{i+1}_evolution.jpg', format='jpeg', dpi=300)
    plt.show()
