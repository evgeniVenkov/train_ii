import pandas as pd
import numpy as np

l = np.ones((3,3))

df = pd.DataFrame(l,index = ['A','B','C'])


# Создайте DataFrame.
dict_in = {'age':[88,99,13],
            'стаж работы': [1, 2, 3],
           'зарплата': [4, 5, 6]}


df = pd.DataFrame(dict_in)

new_df = df * [1,12,1000]
new_df['age'] += 1900

print(new_df)
