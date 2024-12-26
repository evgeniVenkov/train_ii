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

products = ['product_a', 'product_b', 'product_c', 'product_d']
product = np.random.choice(products, size=100)
product_price = np.random.randint(0, 6, size=100) * 870
count = np.random.randint(1, 6, size=100)
buy = np.random.choice([True, False], size=100)
df = pd.DataFrame({'product': product, 'product_price': product_price, 'count': count, 'buy': buy})

def func(row):

    if row['product_price']< 1500:
        count = 5
    elif row['product_price']< 3000:
        count = 3
    else:
        count = 1
    return count
def func1(ser):

    if ser < 1500:
        ser =  'Покупать'
    elif ser <= 3000:
        ser = 'Есть над чем подумать'
    else:
        ser = 'И думать нечего, не брать'

    return ser
def func2(ser):
    if ser == True:
        ser = 'Оплачено'
    else:
        ser = 'Не оплачено'

    return ser

def func3(row, prod = None):
    if prod == None:
        if row['buy'] == True:
            return row['product_price'] * row['count']

        else: return None
    elif row['product'] == prod:
        return row['product_price'] * row['count']
    return None


df['price'] = df.apply(func3 , axis=1)


df = pd.read_csv('stud.csv')
count_tru = 50 - df.set_index('name').sum(axis=1)
count_nan = 50 - df.set_index('name').count(axis= 1)
new_ser = count_tru -count_nan
new_ser = new_ser.astype('int64')

correct = df.set_index("name").isin([0])

new_df = correct.agg([np.sum, np.mean])

df.fillna(value= 0, inplace=True)
new_df = df.agg(sum_solved=("5",np.sum),mean_solved=("5",np.mean))
print(new_df)


