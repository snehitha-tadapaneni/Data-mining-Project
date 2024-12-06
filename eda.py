#%%

# EDA
# Different methods of committing crimes (GUN, KNIFE, others)

# Importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%

cp_data = pd.read_csv("final_return_new.csv")

#%%
# Drop rows where the 'price' column is missing
cp_data = cp_data.dropna(subset=['price'])

# Drop columns that contain missing values: 'num_units' and 'kitchens'
cp_data.dropna(axis=1, inplace=True)
#%%
# Let's Check again
print(cp_data.info())

#%%
# Renaming the columns, all to lower cases
cp_data.columns = cp_data.columns.str.lower()

cp_data.isna().sum()

cp_data.head()

#%%
# Drop the 'sale_year' column
# we will also drop the 'total_gross_column' as we can rely on the median income values for our analysis
cp_data = cp_data.drop(columns=['saledate', 'start_year', 'unnamed: 0', 'total_gross_income'])

# Rename the 'saledate' column to 'year'
cp_data = cp_data.rename(columns={'saleyear': 'year'})

#%%
# Convert all the float to int
#######################Add ur code here if u any

#%%
# Converting ward object type to int
# Remove 'Ward ' prefix and convert to integer
cp_data['ward'] = cp_data['ward'].str.replace('Ward ', '', regex=True).astype(int)

#%%
cp_data_cleaned = cp_data.copy()

q1, q3 = np.percentile(cp_data['price'], 25), np.percentile(cp_data['price'], 75)

iqr = q3-q1
lower = q1 - 1.5*iqr
upper = q3 + 1.5*iqr

# Removing the outliers
cp_data_cleaned = cp_data_cleaned[(cp_data_cleaned['price'] >= lower) & (cp_data_cleaned['price'] <= upper)]

print("New Shape: ", cp_data_cleaned.shape)
cp_data_cleaned.head()
#%%
# Our final cleaned data has columns
print(cp_data_cleaned.columns)

#%%
# Assuming 'df' is your DataFrame
print(cp_data_cleaned[['price', 'method_gun', 'method_knife', 'method_others']].describe())


sns.boxplot(data=cp_data_cleaned[['method_gun', 'method_knife', 'method_others']])
plt.title('Box Plot of Crime Methods')
plt.show()


sns.histplot(data=cp_data_cleaned[['method_gun', 'method_knife', 'method_others']], kde=True)
plt.title(f'Histogram of Crime Methods')
plt.xlabel('Count of Crimes')
plt.ylabel('Frequency')
plt.show()


#%%

##### Bivariate Analysis #######

df_corr =cp_data_cleaned.corr().sort_values(by = 'price', ascending = False)[['price']]

df_corr

#%%

# fireplace, bathrm, rooms, method_GUN, offense_ASSAULT W/DANGEROUS WEAPON, method_KNiFE, are 

cp_data_cleaned

['bathrm', 'rooms', 'bedrm', 'price', 'fireplaces', 'census_tract',
       'ward', 'year', 'median_gross_income', 'offense_arson',
       'offense_assault w/dangerous weapon', 'offense_burglary',
       'offense_homicide', 'offense_motor vehicle theft', 'offense_robbery',
       'offense_sex abuse', 'offense_theft f/auto', 'offense_theft/other',
       'method_gun', 'method_knife', 'method_others', 'shift_day',
       'shift_evening', 'shift_midnight']
#%%

correlation = cp_data_cleaned[['price', 'method_gun', 'method_knife', 'method_others']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#%%

### Data viz on crime counts using different methods and housing price ####

# No standardization

methods = ['method_gun', 'method_knife', 'method_others']
for method in methods:
    sns.scatterplot(x=cp_data_cleaned[method], y=cp_data_cleaned['price'])
    plt.title(f'Housing Price vs {method}')
    plt.xlabel(method)
    plt.ylabel('Price')
    plt.show()

for method in methods:
    sns.jointplot(x=cp_data_cleaned[method], y=cp_data_cleaned['price'], kind='reg')
    plt.title(f'Housing Price vs {method}')
    plt.show()


# # Log Transform on housing price
# for method in methods:
#     sns.scatterplot(x=cp_data_cleaned[method], y=np.log(cp_data_cleaned['price']))
#     plt.title(f'Standardized Housing Price vs {method}')
#     plt.xlabel(method)
#     plt.ylabel('Price')
#     plt.show()
# for method in methods:
#     sns.jointplot(x=cp_data_cleaned[method], y=np.log(cp_data_cleaned['price']), kind='reg')
#     plt.title(f'Housing Price vs {method}')
#     plt.show()

# %%

from scipy.stats import spearmanr

for method in methods:
    corr_coef, p_value = spearmanr(cp_data_cleaned[method], cp_data_cleaned['price'])
    print(f'{method}: Spearman Correlation = {corr_coef:.4f}, p-value = {p_value:.4e}')

# %%

import statsmodels.api as sm

for method in methods:
    X = sm.add_constant(cp_data_cleaned[method])
    y = cp_data['price']
    model = sm.OLS(y, X).fit()
    print(f'Regression Results for {method}')
    print(model.summary())


#%%

X = sm.add_constant(cp_data_cleaned[methods])
y = cp_data['price']
model = sm.OLS(y, X).fit()
print('Multiple Linear Regression Results')
print(model.summary())


#%%

from statsmodels.stats.outliers_influence import variance_inflation_factor

control_vars = ['bathrm', 'rooms','fireplaces']

X_variables = cp_data_cleaned[methods + control_vars]
X_with_const = sm.add_constant(X_variables)
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
vif['Variable'] = X_with_const.columns
print(vif)
#%%

## House Physical Features EDA ###


# Univariate Analysis

# Categorical Features

## Census Tract

cp_data_cleaned
#%%

census_counts = cp_data_cleaned['census_tract'].value_counts()
# plt.figure(figsize=(10, 5))
# census_counts.plot(kind='bar')
# plt.title('Census Tract Frequency Distribution')
# plt.xlabel('Census Tract')
# plt.ylabel('Count')
# plt.show()


import squarify

# treemap of census tract frequencies
plt.figure(figsize=(12, 8))
squarify.plot(sizes=census_counts.values, label=census_counts.index, alpha=0.8)
plt.title('Census Tract Frequency Treemap')
plt.axis('off')
plt.show()

#%%

# Year of sold
census_counts = cp_data_cleaned['year'].value_counts()
plt.figure(figsize=(10, 5))
census_counts.plot(kind='bar')
plt.title('Year of Sale Frequency Distribution')
plt.xlabel('Year of Sale')
plt.ylabel('Count')
plt.show()

#%%
# Numerical Features 
cp_data_cleaned.columns

#%%

# Physical Features
numerical = ['bathrm', 'rooms', 'bedrm', 'price', 'fireplaces']

numerical_features = ['bathrm', 'rooms', 'fireplaces', 'year']

target = ['price']

print(cp_data_cleaned[numerical].describe())
print(f"Skewness: {cp_data_cleaned[numerical].skew()}")
print(f"Kurtosis: {cp_data_cleaned[numerical].kurt()}")


#%%
cat_features = ['census_tract', 'ward']
cat_features
#%%
# data viz 

cp_data_cleaned[numerical].hist(bins=30, figsize=(15, 10))
plt.suptitle('Histogram of Numerical Features')
plt.show()

plt.figure(figsize=(16, 12))

for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)  # Create a grid with 2 rows and 3 columns
    sns.boxplot(y=cp_data_cleaned[feature])
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
cp_data_cleaned.boxplot(column='price')  # Use 'column' argument with the column name as a string
plt.title('Boxplot for Price')
plt.ylabel('Price')  # Add y-axis label for better clarity
plt.show()
#%%

# Bivariate Analysis

## Numerical vs Numerical

import seaborn as sns

# Pairplot for all numerical features
sns.pairplot(cp_data_cleaned[numerical])
plt.suptitle('Pairplot for Numerical Features', y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cp_data_cleaned[numerical].corr(method = 'spearman'), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
# %%

# Numerical features vs Target (price)

import seaborn as sns
import matplotlib.pyplot as plt


for feature in numerical_features:

    # Box plot for dataset without outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=cp_data_cleaned[feature], y=cp_data_cleaned['price'])
    plt.title(f'Price vs. {feature}')
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.show()

# %%
