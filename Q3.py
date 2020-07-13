import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

##################################################

df = pd.read_excel('Online Retail.xlsx', sheetname=0)

print(df.head())
print(set(df['Country']))
print(len(set(df['CustomerID'])))
'''
從資料可知，消費記錄來自38國 139452個消費者。
分析策略上先觀察固定效果(購買時間、地點)的影響，再去分群individual程度的行為。
題目要求分析哪些顧客有相似的消費行為，
我認為可分為前往購物的行為(例如只在周一消費)跟需求彈性的模式(例如頻繁購買某商品或少買高價品)。
據我理解題目要求的是以消費者代號為分類對象，因此單一購物者若有兩種消費模式也不能視為兩種分類型態。
'''
# Data整理
df = df[df['Quantity'] >= 0]

# 建立購買時間變數
df['date'] = df['InvoiceDate'].map(lambda x: x.date())
df['month'] = df['InvoiceDate'].map(lambda x: x.month)
df['weekday'] = df['InvoiceDate'].map(lambda x: x.weekday())
df['weekend'] = df['weekday'].map(lambda x: 1 if x >= 5 else 0)
df['morning'] = df['InvoiceDate'].map(lambda x: 1 if x.hour < 12 else 0)
df['afternoon'] = df['InvoiceDate'].map(lambda x: 1 if (x.hour >= 12) & (x.hour < 18) else 0)
df['night'] = df['InvoiceDate'].map(lambda x: 1 if x.hour > 18 else 0)

# 以簡單線性迴歸觀察
# 這部分想觀察單筆商品消費金額與溝買時間的關係，但解釋性相當低
df['cost'] = df['Quantity'] * df['UnitPrice']
df = pd.get_dummies(df, columns=['month'])
X = df[['weekend', 'morning', 'night'] + ['month_{}'.format(m) for m in range(1,12)]]
Y1 = df['cost']
reg1 = LinearRegression().fit(X, Y1)
print('coef={}, socre={}'.format(reg1.coef_, reg1.score(X,Y1)))

# 以單次消費行為為觀察對象
df_1 = df.groupby('InvoiceNo').sum().reset_index()
for time_var in ['weekend', 'morning', 'night'] + ['month_{}'.format(m) for m in range(1,13)]:
    df_1[time_var] = df_1[time_var].map(lambda x: 1 if x > 0 else 0)
# df_1 = df_1[['Quantity', 'UnitPrice', 'CustomerID', 'weekend', 'afternoon', 'night', 'morning', 'cost']].reset_index()
df_n = df.groupby('InvoiceNo').count().reset_index()
df_1 = pd.merge(df_1, df_n.rename(columns={'StockCode': 'N'})[['InvoiceNo', 'N']], how='inner', on='InvoiceNo')
df_1['weekday'] = df_1.apply(lambda x: x['weekday']//x['N'], axis=1)
df_1['avg_q_cost'] = df_1.apply(lambda x: x['cost']//x['Quantity'], axis=1)

# 以線性模型建模，時間變數對單次消費行為解釋性上升，但仍相當低
X = df_1[['weekend', 'morning', 'night']+ ['month_{}'.format(m) for m in range(1,12)]]
Y1 = df_1['cost']
reg1 = LinearRegression().fit(X, Y1)
print('coef={}, socre={}'.format(reg1.coef_, reg1.score(X,Y1)))

# Feature Engineering
# 建立敘述統計的Feature以捕捉個別消費者的行為
df_std = df.groupby('InvoiceNo').std().reset_index()
df_1 = pd.merge(df_1, df_std.rename(columns={'cost': 'cost_std', 'Quantity':'Quantity_std', 'UnitPrice':'UnitPrice_std'})[['InvoiceNo', 'cost_std','Quantity_std','UnitPrice_std']], how='inner', on='InvoiceNo')
for col in ['cost_std','Quantity_std','UnitPrice_std']:
    df_1[col] = df_1[col].fillna(0)
df_ind = df_1.groupby('CustomerID').mean().reset_index()
df_ind = df_ind[df_ind['CustomerID'] != 0.0]

# 消費時間的分群
fig = plt.Figure()
plt.scatter(x=df_ind['night'], y=df_ind['morning'])
plt.xlabel('night_proba')
plt.ylabel('morning_proba')
plt.show()

print(len(df_ind[df_ind['night'] == 1])/len(df_ind))
print(len(df_ind[df_ind['afternoon'] == 1])/len(df_ind))
print(len(df_ind[df_ind['morning'] == 1])/len(df_ind))
print(len(df_ind[df_ind['night'] == 0])/len(df_ind))
print(len(df_ind[df_ind['afternoon'] == 0])/len(df_ind))
print(len(df_ind[df_ind['morning'] == 0])/len(df_ind))
print(len(df_ind[(df_ind['morning'] == 1) & (df_ind['weekend'] == 0)])/len(df_ind))
'''
由統計可知，每次都在早上/下午/晚上消費的消費者分別佔30.5%、3%、0.7%，
其中每次在平日早上購物的消費者佔27.3%，這群消費者中包含某些特定族群的機率可能較高，例如家庭主婦或自由業者
若我們認為這些族群跟其他族群的消費模式有顯著不同，則這就是一個可能的分類方式
'''
df_g1 = df_ind[(df_ind['morning'] == 1) & (df_ind['weekend'] == 0)]
df_g2_1 = df_ind[(df_ind['morning'] != 1)]
df_g2_2 = df_ind[df_ind['weekend'] != 0]
df_g2 = pd.concat([df_g2_1, df_g2_2], axis=0)
df_g2 = df_g2.groupby('CustomerID').mean()


def distribution_show(df_i, group_name='g1'):
    print('Group {}'.format(group_name))
    print('mean={}, stddev={}, median={}'.format(np.mean(df_i['cost']), np.std(df_i['cost']),
                                                         np.median(df_i['cost'])))
    plt.hist(df_i['cost'], color='blue', edgecolor='black', bins=int(180 / 5))
    plt.show()

distribution_show(df_g1, 'g1')
distribution_show(df_g2, 'g2')
# group 1 mean=587.6463141614906, stddev=3105.9930546408345, median=300.9949999999999
# group 2 mean=445.9799909337447, stddev=846.7783528620998, median=300.9949999999999
'''
若以平均單次消費金額為分類對象，
可以看出在平日早上購物族群(Group 1) 與其他(Group 2) 的分布平均跟標準差明顯有不同，
但中位數是差不多的，因此以機率密度圖做比較，
可知Group 1中有離群值，99.6%的 消費者均消費在10000以下 (Group 2中則是99.8%)，
因此將離群值取出後，這個分群對於平均單次的消費並無重大意義。
'''
distribution_show(df_g1[df_g1['cost'] < 10000], 'g1-2')

df_afternoon = df_ind[df_ind['afternoon'] == 1]
df_night = df_ind[df_ind['night'] == 1]
distribution_show(df_afternoon, 'g-afternoon')
distribution_show(df_night, 'g-afternoon')

'''
小結：
由於時間關係本題未完成，
目前已知是若以平均單次消費金額為分類目標，
全下午消費者與全晚間購物型消費者或許可以分別成為類組，
而以早上購物去區分消費者的效果不彰。
接下來會以購物種類、消費頻率去做消費者的分類。
'''