The scripts shown include various data algorithms which can be used in sequence to undergo a Dynamic Mode Decomposition (DMD) on a given time series dataset:

The standard DMD algorithm aims to construct a least squares estimate of a finite dimensional Koopman Operator whose eigenvectors (the DMD modes) which are temporally invariant,
can be extracted and used to propagate the state through time. The script utilises several algorithms to accomplish this:
* Principal Component Analysis (PCA):\
By seeking a basis in which the Covariance matrix of the Data matrix is diagonal, we can transform to a new state space where all variables are uncorrelated. This is done by standard
diagonalisation methods as the Covariance matrix is square by design [1](https://en.wikipedia.org/wiki/Principal_component_analysis). In this step, the data is also centred to account
for any systematic shift.
* SVD:\
The singular value decomposition is used (`numpy.linalg.svd` returns all 3 relevant decomposition components) to construct Moore-Penrose pesudo-inverses and identify the singular
values of the finite dimensional Koopman operator.
* Hankel or Delay Embedding DMD:\
For Data matrices (nxm) where n << m or m << n, the built-in numpy `linalg.svd` method with the `full_matrices=False` argument results in a siginificant loss of information
by truncating the input matrices to be nxs for s = min(n,m). Refer to the relevant repository [2](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) for
further information. By concatenating columns of the original data matrix, we can expand the dimensionality of the system which increases computational expense but results in
longer time series estimation accuracy [3](https://www.mdpi.com/2227-7390/12/5/762) and works around the truncation imposed by the `linalg.svd` method.
* DMD Mode propagation:\
Both continuous and discrete time propagations of the DMD modes can be made [4](https://arxiv.org/abs/2102.12086) whose predictivity can be visualised through graphing the evolution
of each state variable. 

The algorithm can be tested with the `test_data.csv` time series (the results for Hankel s=50 are shown in the attached jpg files)
