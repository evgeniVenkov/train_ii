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


df = pd.read_csv('data/stud.csv')
count_tru = 50 - df.set_index('name').sum(axis=1)
count_nan = 50 - df.set_index('name').count(axis= 1)
new_ser = count_tru -count_nan
new_ser = new_ser.astype('int64')

correct = df.set_index("name").isin([0])

new_df = correct.agg([np.sum, np.mean])

df.fillna(value= 0, inplace=True)
new_df = df.agg(sum_solved=("5",np.sum),mean_solved=("5",np.mean))
# --------------------------------------------------------------------------------------------------
def getdf():
    N_client = 10
    N_product = 15

    client = {i: f'client_{i}' for i in range(N_client)}
    product = {i: f'product_{i}' for i in range(N_product )}
    price = {i: (i+1)*1000 for i in range(N_product)}

    client_list = np.random.randint(0, N_client, 100)
    product_list = np.random.randint(0, N_product , 100)

    df = pd.DataFrame({'client': client_list,
                       'product': product_list,
                       'price': product_list}).replace({'client': client, 'product': product, 'price': price})
    return df

df = getdf()
new_ser = df.groupby('client')['price'].sum()
new_ser = df.groupby('client')['product'].count()
new_ser1 = df['client'].value_counts().sort_index()

new_df = df.groupby(['client', 'product']).size().reset_index(name='count')

new_df = df.groupby('product').agg(count=('product', 'size'),
                                   sum_price=('price', 'sum'))

new_ser = df.groupby('product')['client'].unique()

new_df = df.groupby('client').apply(
    lambda group: pd.Series({'product_2': (group['product'] == 'product_2').sum()})
)


new_df = df.groupby('client').apply(
    lambda group: pd.Series({
        'product_3': (group['product'] == 'product_3').sum(),
        'sum_price': group.loc[group['product'] == 'product_3', 'price'].sum()
    })
)

df = pd.read_csv('data/users.csv')
pc_users = df[df['device'] == 'PC']['user'].unique()
user_list = [user for user in pc_users if all(df[(df['user'] == user)]['device'] == 'PC')]

user_list = df.groupby('user')['device'].nunique()
user_list = user_list[user_list >= 3].index.tolist()

df = pd.read_csv('data/users_action.csv')

new_df = df[df['device'] == 'phone']
new_df = new_df.groupby('user')['action'].apply(lambda x: (x == True).sum()).reset_index(name='action_True')
new_df.set_index('user', inplace=True)


# ---------------------------------------------------------------------------------------
def get_new_df():
    N_client = 10
    N_product = 15

    client_1 = {i: f'client_{i}' for i in range(N_client)}
    client_2 = {i: f'client_{i}' for i in range(3, N_client + 3)}

    data_product = {i: f'product_{i}' for i in range(N_product)}
    price_1 = {i: (i + 1) * 1000 for i in range(15)}
    price_2 = {i: (i + 1) * 900 for i in range(15)}

    client_list_1 = np.random.randint(0, N_client, 100)
    client_list_2 = np.random.randint(3, N_client + 3, 100)

    product_list = np.random.randint(0, N_product, 100)

    shop_1 = pd.DataFrame({'client': client_list_1,
                           'product': product_list,
                           'price': product_list}).replace(
        {'client': client_1, 'product': data_product, 'price': price_1})
    shop_2 = pd.DataFrame({'client': client_list_2,
                           'product': product_list,
                           'price': product_list}).replace(
        {'client': client_2, 'product': data_product, 'price': price_2})
    product = pd.DataFrame({'product': data_product.values(),
                            'color': np.random.choice(['black', 'red', 'green', 'white', 'blue'], 15),
                            'weight': np.random.randint(10, 20, 15) * 10})
    return shop_1, shop_2, product

df_shop_1, df_shop_2, df_product = get_new_df()

print(df_shop_1.head())

dates = pd.date_range(start='2020-01-01', end='2025-01-01', periods=100)
income = np.random.randint(1000, 10000, size=100)

df = pd.DataFrame({
    'date': dates,
    'salary': income
})

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')
print(df.head())

new_df = df.groupby('month')['salary'].sum().reset_index()
new_df['month'] = pd.to_datetime(new_df['month'].astype(str), format='%Y-%m').dt.month
print(new_df.head())