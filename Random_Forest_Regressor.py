import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.stats import spearmanr
from scipy.stats import f_oneway


df = pd.read_csv('final_return_new.csv')
df.describe()


sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
plt.title("Missing values heatmap")
plt.show()
df = df.drop(columns=['saledate', 'start_year', 'unnamed: 0','total_gross_income'])
df = df.rename(columns={'saleyear':'year'})

df = df.dropna(subset=['price'])

df.dropna(axis=1, inplace=True)

df['ward'] = df['ward'].str.replace('Ward ', '', regex=True).astype(int)

df.columns = df.columns.str.lower()
num_cols = ['bathrm', 'rooms', 'fireplaces', 'bedrm', 'year', 'ward', 'median_gross_income', 'offense_arson', 'offense_assault w/dangerous weapon', 'offense_burglary', 'offense_homicide', 'offense_motor vehicle theft', 'offense_robbery', 'offense_sex abuse', 'offense_theft f/auto', 'offense_theft/other', 'method_gun', 'method_knife', 'method_others', 'shift_day', 'shift_evening', 'shift_midnight']

df[num_cols].hist(figsize=(10,20),layout=(6,4),edgecolor='black')
plt.suptitle("Distributions of Numerical Features")
plt.show()
plt.figure(figsize=(13,8))
sns.histplot(df['price'],kde=True)
plt.title("Distribution of Housing Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x=df['price'])
plt.title("Boxplot Of housing prices")
plt.show()

# Removing the outliers from the target variable: price

## Performing EDA and statistical tests

import numpy as np
df1 = df.copy()

q1, q3 = np.percentile(df['price'],25), np.percentile(df['price'],75)

iqr = q3 - q1
lower = q1 - 1.5*iqr
upper = q3 + 1.5*iqr

df1 = df1[(df1['price']>=lower) & (df1['price']<=upper)]

print("New Shape ",df1.shape)
print(f"Old shape {df.shape} new Shape {df1.shape}")
# Boxplot of housing prices post outliers removal

plt.figure(figsize=(10,6))
sns.boxplot(x=df1['price'])
plt.title("Boxplot Of housing prices")
plt.show()
# Understanding the methods used and the frequency

methods = ['method_gun', 'method_knife', 'method_others']
method_sum = df1[methods].sum()

plt.figure(figsize=(10,6))
method_sum.plot(kind='bar',color=['skyblue', 'orange','green'],alpha=0.8)
plt.title('Frequency of Each Method Type', fontsize=14)
plt.xlabel('Method Type', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=0)
plt.show()
plt.figure(figsize=(8, 6))
sns.histplot(df1['price'], kde=True, color='purple', bins=30)
plt.title('Distribution of Price', fontsize=14)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Price vs year
sns.boxplot(data=df1,y=df1['price'],x=df1['year'])
numerical_cols = df1.select_dtypes(include=['float64', 'int64']).columns
numerical_df = df1[numerical_cols]

corr = numerical_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title('Correlation Heatmap')
plt.show()
plt.figure(figsize=(8,12))
heatmap = sns.heatmap(numerical_df.corr()[['price']].sort_values(by='price', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')

# Price vs Rooms

plt.figure(figsize=(10, 6))
sns.boxplot(x='rooms', y='price', data=df1)
plt.title("Housing Prices Based on Number of Rooms")
plt.xlabel("Number of Rooms")
plt.ylabel("Price")
plt.show()
# Price vs bedrooms

plt.figure(figsize=(10, 6))
sns.boxplot(x='bathrm', y='price', data=df1)
plt.title("Housing Prices Based on Number of Bathrooms")
plt.xlabel("Number of Bathrooms")
plt.ylabel("Price")
plt.show()
# Prices vs bed room
plt.figure(figsize=(10, 6))
sns.boxplot(x='bedrm', y='price', data=df1)
plt.title("Housing Prices Based on Number of Bed Rooms")
plt.xlabel("Number of Rooms")
plt.ylabel("Price")
plt.show()
# Prices vs ward
plt.figure(figsize=(10, 6))
sns.barplot(x='ward', y='price', data=df1)
plt.title("Housing Prices Based on Number of ward")
plt.xlabel("Ward")
plt.ylabel("Price")
plt.show()
## Scatter plot: All method types(gun, knife, others) vs housing prices

# Prepare data for scatter plot
plt.figure(figsize=(10, 6))

# Plot for METHOD_GUN
plt.scatter(
    df1.loc[df1['method_gun'] == 1, 'price'],
    df1.loc[df1['method_gun'] == 1].index,
    color='red', label='Gun', alpha=0.6
)

# Plot for METHOD_KNIFE
plt.scatter(
    df1.loc[df1['method_knife'] == 1, 'price'],
    df1.loc[df1['method_knife'] == 1].index,
    color='blue', label='Knife', alpha=0.6
)

# Plot for METHOD_OTHERS
plt.scatter(
    df1.loc[df1['method_others'] == 1, 'price'],
    df1.loc[df1['method_others'] == 1].index,
    color='yellow', label='Others', alpha=0.6
)
df1.head()
# Calculate Spearman correlation between 'price' and each 'method' type
corr_gun, p_gun = spearmanr(df1['price'],df1['method_gun'])
corr_knife,p_knife = spearmanr(df1['method_knife'],df1['method_knife'])
corr_others, p_others = spearmanr(df1['price'], df1['method_others'])

print(f"Spearman Correlation for method_GUN: {corr_gun}, p-value: {p_gun}")
print(f"Spearman Correlation for method_knife: {corr_knife}, p-value: {p_knife}")
print(f"Spearman Correlation for method_GUN: {corr_others}, p-value: {p_others}")

<!-- # Let us state our hypothesis,<br>
# Null Hypothesis (H₀): There is no monotonic relationship between price and method_GUN, method_KNIFE, method_OTHERS.
# Alternative Hypothesis (H₁): There is a significant monotonic relationship between price and method_GUN, method_KNIFE, method_OTHERS.

# 1. For method_GUN:<br>
# Interpretation: Since the p-value is 0.0 (which is less than 0.05), we reject the null hypothesis, indicating a significant monotonic relationship between price and method_GUN. This suggests that as the price increases or decreases, there is a tendency for the frequency of gun-related incidents to change in a monotonic manner.

# 2. For method_KNIFE:<br>
# Interpretation: The p-value is also 0.0, which is less than 0.05, so we reject the null hypothesis, indicating a significant monotonic relationship between price and method_KNIFE. This suggests that there is a weak but significant trend of knife-related incidents associated with price changes.

# 3. For method_OTHERS:<br>
# Interpretation: Since the p-value is 0.976 (which is greater than 0.05), we fail to reject the null hypothesis, indicating no significant monotonic relationship between price and method_OTHERS. This suggests that changes in price do not significantly affect the occurrence of incidents categorized as "Others."
#<br> -->
## Let us state our hypothesis,<br>
# Null Hypothesis (H₀): There is no monotonic relationship between price and method_GUN, method_KNIFE, method_OTHERS.
Alternative Hypothesis (H₁): There is a significant monotonic relationship between price and method_GUN, method_KNIFE, method_OTHERS.

1. For method_GUN:<br>
Interpretation: Since the p-value is 0.0 (which is less than 0.05), we reject the null hypothesis, indicating a significant monotonic relationship between price and method_GUN. This suggests that as the price increases or decreases, there is a tendency for the frequency of gun-related incidents to change in a monotonic manner.

2. For method_KNIFE:<br>
Interpretation: The p-value is also 0.0, which is less than 0.05, so we reject the null hypothesis, indicating a significant monotonic relationship between price and method_KNIFE. This suggests that there is a weak but significant trend of knife-related incidents associated with price changes.

3. For method_OTHERS:<br>
Interpretation: Since the p-value is 0.976 (which is greater than 0.05), we fail to reject the null hypothesis, indicating no significant monotonic relationship between price and method_OTHERS. This suggests that changes in price do not significantly affect the occurrence of incidents categorized as "Others."
<br>
## Scatter plot between crime categories vs the price distribution

## Aggregate crime counts as violent crime and property crime
df1['violent_crime_count'] = df1[['offense_assault w/dangerous weapon', 'offense_homicide', 'offense_robbery', 'offense_sex abuse']].sum(axis=1)

df1['property_crime_count'] = df1[['offense_arson', 'offense_burglary', 'offense_motor vehicle theft', 'offense_theft f/auto', 'offense_theft/other']].sum(axis=1)

plt.figure(figsize=(10, 6))
plt.scatter(df1['violent_crime_count'], df1['price'], color='red', alpha=0.6, label='Violent Crimes')
plt.scatter(df1['property_crime_count'], df1['price'], color='blue', alpha=0.6, label='Property Crimes')
plt.title('Scatter Plot: Violent and Property Crimes vs Price')
plt.xlabel('Crime Count')
plt.ylabel('Price')
plt.legend()
plt.grid(alpha=0.5)
plt.show()
### As, we can see the above scatter plot is too complex to understand.
### Let us aggregate the data based on census tract and plot the violent and property crime values for more clarity.
# Aggregate data by census tract
tract_data = df1.groupby('census_tract').agg({
    'violent_crime_count': 'sum',
    'property_crime_count': 'sum',
    'price': 'mean'  # Average price per tract
}).reset_index()

plt.figure(figsize=(10, 6))
plt.scatter(tract_data['violent_crime_count'], tract_data['price'], color='red', alpha=0.6, label='Violent Crimes')
plt.scatter(tract_data['property_crime_count'], tract_data['price'], color='blue', alpha=0.6, label='Property Crimes')
plt.title('Crime Counts vs Average Price by Census Tract')
plt.xlabel('Aggregated Crime Count')
plt.ylabel('Average Price ($)')
plt.legend()
plt.grid(alpha=0.5)
plt.show()
### Compute correlation coefficients: Price vs violent crime vs property crime
correlation_matrix = df1[['price', 'violent_crime_count', 'property_crime_count']].corr()

# Display correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Price vs Crime Counts')
plt.show()
df1['ward'] = df1['ward'].astype(str)

#%%
# Comparing prices based on the crimes in each ward
grouped_data = df1.groupby('ward').agg({
    'price': 'mean',
    'violent_crime_count': 'sum',
    'property_crime_count': 'sum'
}).reset_index()
# Set the plotting style
sns.set_style("whitegrid")

# Initialize the figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

# First bar plot: Ward vs Price
sns.barplot(data=grouped_data, x='ward', y='price', ax=axes[0], palette='coolwarm')
axes[0].set_title('Ward vs Average Price')
axes[0].set_xlabel('Ward')
axes[0].set_ylabel('Average Price')

# Second bar plot: Ward vs Crime Counts (stacked with Violent and Property Crimes)
melted_crime_data = grouped_data.melt(
    id_vars=['ward'], value_vars=['violent_crime_count', 'property_crime_count'],
    var_name='Crime Type', value_name='Crime Count'
)
sns.barplot(data=melted_crime_data, x='ward', y='Crime Count', hue='Crime Type', ax=axes[1], palette='viridis')
axes[1].set_title('Ward vs Crime Counts')
axes[1].set_xlabel('Ward')
axes[1].set_ylabel('Total Crime Counts')
axes[1].legend(title='Crime Type')

# Adjust layout
plt.tight_layout()
plt.show()
# Model Building
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

X = df1[['bathrm','rooms', 'bedrm','median_gross_income',
       'fireplaces', 'census_tract', 'ward', 'year','violent_crime_count','property_crime_count',
       'method_gun', 'method_knife', 'method_others', 'shift_day',
       'shift_evening', 'shift_midnight']]
y = df1['price']
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

#%%
print(f"Test MSE: {mse}")
print(f"Test RMSE: {np.sqrt(mse)}")
print(f"Test R2 Score: {r2}")
feature_importance = pd.DataFrame({
    'features':X.columns,
    'importance' : model.feature_importances_
}).sort_values('importance', ascending=True)

#%%

print("Features Importance")
print(feature_importance.sort_values('importance',ascending=False))
from sklearn.model_selection import cross_val_score

cv_scores_r2 = cross_val_score(model,X,y,cv=5,scoring='r2')
cv_scores_rmse = cross_val_score(model,X,y,scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores_rmse)
from scipy import stats

def qq(residual):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    stats.probplot(residual, dist="norm", plot=ax)
    ax.set_title('QQ Plot')
    plt.show()

#%%

residual = y_test - y_pred

qq(residual)
feature_importance = feature_importance.sort_values
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['features'], feature_importance['importance'])
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')

# Add grid for better readability
plt.grid(True, axis='x', linestyle='--', alpha=0.6)

# Tight layout to prevent label cutoff
plt.tight_layout()

plt.show()
