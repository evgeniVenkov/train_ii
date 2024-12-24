import pandas as pd
import numpy as np


# Создайте DataFrame.
dict_in = {'age':[88,99,13,22,17,36],
            'стаж работы': [1, 2, 3,None,5,6],
           'зарплата': [40000, 50000, 60000,33000,27000,11000]}


df = pd.DataFrame(dict_in)

df.dropna()
#поиск и замена нан
df.fillna(value={"age":1,
                 "стаж работы":"NO"}, inplace=True)

df.query("`зарплата` > 35000")

