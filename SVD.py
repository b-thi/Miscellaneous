## Setting wd
current_dir = getcwd()
chdir(current_dir)


# Singular-value decomposition
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd

# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)

# SVD
U, s, VT = svd(A)
print(U)
print(s)
print(VT)

##### Now getting back original matrix

# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))

# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = diag(s)

# reconstruct matrix
B = U.dot(Sigma.dot(VT))

# Plotting    
print(B)
