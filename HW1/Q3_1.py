import numpy as np
import random
D = 3
K = 5


X = np.full((D,1),1)
for x_ in range(X.shape[0]):
    X[x_] = x_ + 1

W = np.full((K,D),0.3)
for x_ in range(W.shape[0]):
    for y_ in range(W.shape[1]):
        W[x_,y_] = x_*W.shape[1] + y_

print("X:\n",X)
print("W:\n",W)

W_dot_X = W.dot(X)
print("W.X:\n",W_dot_X)

pow_w_dot_x = np.power(W_dot_X,2)
print("g(W.X):\n",pow_w_dot_x)

print("\n")
print(np.power(W,2).dot(np.power(X,2)))