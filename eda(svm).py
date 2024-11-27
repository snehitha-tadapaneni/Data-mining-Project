#%%
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
#%%
df = pd.read_csv('/Users/sayam_palrecha/Desktop/VS/DM_code/Data_Part2.csv')
df
#%%
'''
Multicollinearity occurs when two or more independent variables in a regression model are highly correlated with each other.
This statistical phenomenon can lead to unreliable and misleading results,
making it difficult to determine how individual variables affect the dependent variable
'''
#%%

# Features affecting the price of a property
#%%
corr_matrix1 = df.iloc[:,0:6].corr(method='pearson')

sns.heatmap(corr_matrix1,annot=True,cmap='coolwarm')
#%%

# Offense  to price

df1 = df[['price','offense_ARSON', 'offense_ASSAULT W/DANGEROUS WEAPON',
       'offense_BURGLARY', 'offense_HOMICIDE', 'offense_MOTOR VEHICLE THEFT',
       'offense_ROBBERY', 'offense_SEX ABUSE', 'offense_THEFT F/AUTO',
       'offense_THEFT/OTHER']]

corr_matrix2 = df1.corr(method='pearson')

sns.heatmap(corr_matrix2,annot=True,cmap='coolwarm')
#%%

df2 = df[['price','offense_ARSON','method_GUN', 'method_KNIFE', 'method_OTHERS']]

corr_matrix2 = df2.corr()

sns.heatmap(corr_matrix2,annot=True,cmap='coolwarm')
#%%

df.iloc[:,1:16]

from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df.iloc[:,1:16]

vif_data = pd.DataFrame()

vif_data["feature"] = X.columns

vif_data['VIF'] = [variance_inflation_factor(X.values,i)
                   for i in range(len(X.columns))]

vif_data

sns.kdeplot(vif_data,fill=True)

# %%
df.columns
# %%
