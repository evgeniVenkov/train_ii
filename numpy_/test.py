import numpy as np

cords =np.empty([10,4,2])
new_cords = np.empty([10,2])

print(cords.shape)
print(new_cords.shape)

new_arr = np.concatenate((cords, new_cords[:, None, :]), axis=1)

print(new_arr.shape)

coefs=np.ones([1,3])
print(coefs.shape)

