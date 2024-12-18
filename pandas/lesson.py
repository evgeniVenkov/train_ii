import pandas as pd
import numpy as np

l = np.ones((3,3))

df = pd.DataFrame(l,index = ['A','B','C'])


# Создайте DataFrame.
dict_in = {'col_1': [1, 2, 3],
           'col_2': [4, 5, 6],
           'col_3': [7, 8, 9],
           'col_4': [10, 11, 12]}

df = pd.DataFrame(dict_in)[['col_1', 'col_3']]
print(df)


