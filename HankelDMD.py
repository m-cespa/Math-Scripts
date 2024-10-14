# Hankel DMD for spatially undersampled data (delay embeddings), s<=n
import numpy as np

data = ...

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