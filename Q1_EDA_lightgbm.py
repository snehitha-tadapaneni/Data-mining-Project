
#%%
# Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import lightgbm as lgb
import warnings
import geopandas as gpd
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler

#%%

##########################
### Data Preprocessing ###
##########################
#%% 
# Load Data
cp_data = pd.read_csv("final_return_new.csv")

# Drop rows where the 'price' column is missing
cp_data = cp_data.dropna(subset=['price'])

# Drop columns that contain missing values: 'num_units' and 'kitchens'
cp_data.dropna(axis=1, inplace=True)

# Let's Check again
print(cp_data.info())

# Renaming the columns, all to lower cases
cp_data.columns = cp_data.columns.str.lower()

cp_data.isna().sum()

cp_data.head()

#%%
# Drop the 'saledate' column
# we will also drop the 'total_gross_column' as we can rely on the median income values for our analysis
cp_data = cp_data.drop(columns=['saledate', 'start_year', 'unnamed: 0', 'total_gross_income'])

# Rename the 'sale_year' column to 'year'
cp_data = cp_data.rename(columns={'saleyear': 'year'})

#%%
# Converting ward object type to int
# Remove 'Ward ' prefix and convert to integer
cp_data['ward'] = cp_data['ward'].str.replace('Ward ', '', regex=True).astype(int)

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

cp_data_cleaned['total_crime_count'] = cp_data_cleaned[
    ['offense_assault w/dangerous weapon', 'offense_homicide', 
     'offense_robbery', 'offense_sex abuse', 'offense_arson',
    'offense_burglary', 'offense_motor vehicle theft', 'offense_theft f/auto', 
    'offense_theft/other']].sum(axis=1)


cp_data_cleaned['violent_crime_count'] = cp_data_cleaned[['offense_assault w/dangerous weapon', 'offense_homicide', 'offense_robbery', 'offense_sex abuse']].sum(axis=1)

cp_data_cleaned['property_crime_count'] = cp_data_cleaned[['offense_arson', 'offense_burglary', 'offense_motor vehicle theft', 'offense_theft f/auto', 'offense_theft/other']].sum(axis=1)

#%% [markdown]

###########
### EDA ###
###########

cp_cleaned = cp_data_cleaned[cp_data_cleaned['fireplaces'] <= 6]
#%% [markdown]

# First, we will define what controlling features to include in our housing price predictive model.
# We will split the features into 

# ## Numerical Features vs Price

num_features = ['bathrm', 'rooms', 'bedrm', 'fireplaces', 'price', 'median_gross_income', 'year']

correlation_matrix = cp_cleaned[num_features].corr(method='spearman')

target_cor = pd.DataFrame(correlation_matrix['price']).sort_values(by = 'price', ascending = False)

target_cor.drop('price', axis = 'index', inplace = True)

plt.figure(figsize=(9, 6))
sns.heatmap(target_cor, annot=True, cmap='Reds', fmt=".2f")

#%%

# Categorical vs Price

# Time vs Price

# Ward vs Price

custom_palette = {
    '1': 'lightblue', '2': 'lightblue', '3': 'lightblue', 
    '4': 'lightblue', '5': 'lightblue', '6': 'lightblue', 
    '7': 'grey', '8': 'grey'
}

plt.figure(figsize=(10, 6))
sns.boxplot(
    x=cp_data_cleaned['ward'].astype(str),  # Convert 'ward' to string to match custom_palette keys
    y=cp_data_cleaned['price'],
    palette=custom_palette
)
plt.title(f'Price vs. Ward')
plt.xlabel('Ward')
plt.ylabel('Price')
plt.show()

#%%

import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('price ~ C(ward)', data=cp_data_cleaned).fit()
anova_table = sm.stats.anova_lm(model, typ=2) 
print('\n===== ANOVA: Ward vs Property Value =====')
print(anova_table) 

#%%
# Time (year)
from scipy.stats import spearmanr

corr_coef, p_value = spearmanr(cp_data_cleaned['year'], cp_data_cleaned['price'])
print(f'Year: Spearman Correlation = {corr_coef:.4f}, p-value = {p_value:.4e}')


model = ols('price ~ C(year)', data=cp_data_cleaned).fit()
anova_table = sm.stats.anova_lm(model, typ=2) 
print('\n===== ANOVA: year vs Property Value =====')
print(anova_table) 
#%%

# Q1: Does neighborhood level crime count affect property value?

# EDA

# Univariate Analysis

# Total Crime Counts
plt.hist(cp_data_cleaned['total_crime_count'], bins=30, edgecolor='black')
plt.xlabel('Median Gross Income')
plt.ylabel('Frequency')
plt.title('Distribution of Total Crime Counts')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Housing Price
plt.hist(cp_data_cleaned['price'], bins=30, edgecolor='black')
plt.xlabel('')
plt.ylabel('Frequency')
plt.title('Distribution of Median Gross Income')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Bivariate Analysis

# Property Price vs Total Crimes
# Scatter plot for total crimes vs price
plt.figure(figsize=(10, 6))
plt.scatter(cp_data_cleaned['total_crime_count'], cp_data_cleaned['price'], alpha=0.6)
plt.title('Scatter Plot: Total Crimes vs Price')
plt.xlabel('Crime Count')
plt.ylabel('Price')
plt.legend()
plt.grid(alpha=0.5)
plt.show()

#%% [markdown]
# In scatter plot, we cannot observe a clear patterns between total crime counts and price.
# To further investigate, we will visualize the distribution of crimes and property values by census tract on a map. 
# This will allow us to explore whether neighborhoods with higher crime occurrences tend to have lower property values.

#%%

# Census Tract

# Aggregate the total crime counts for each neighborhood
tract_crime = cp_data_cleaned[
    ['census_tract', 
     'year', 
     'total_crime_count']
     ].drop_duplicates(subset=['census_tract', 'year']).groupby('census_tract').agg(
            total_crime = ('total_crime_count', 'sum'))

# Aggregate the median house price for each census tract
tract_house = cp_data_cleaned.groupby('census_tract').agg(
    price_median=('price', 'median'), 
    price_mean=('price', 'mean')).reset_index()
tract_house


# Ward

ward_crime = cp_data_cleaned[
    ['ward', 
     'year', 
     'total_crime_count']
     ].drop_duplicates(subset=['ward', 'year']).groupby('ward').agg(
            total_crime = ('total_crime_count', 'sum'))

# Aggregate the median house price for each census tract
ward_house = cp_data_cleaned.groupby('ward').agg(
    price_median=('price', 'median'), 
    price_mean=('price', 'mean')).reset_index()
tract_house

#%%

tract_map = gpd.read_file('Census_Tracts_in_2010.shp')
ward_map = gpd.read_file('/Users/chengyuhan/Downloads/Wards_from_2022/Wards_from_2022.shp') 

# Clean up tract variable and convert into integer
tract_map['TRACT'] = tract_map['TRACT'].str.lstrip('0').astype(int)

# Data Visualization of Crime Counts by Census Tract on Map
crime_merged_map = tract_map.merge(tract_crime, left_on='TRACT', right_on='census_tract', how='left')
crime_merged_map['crime_rate'] = crime_merged_map['total_crime']/ crime_merged_map['P0010001']
crime_merged_map['crime_density'] = crime_merged_map['total_crime']/ crime_merged_map['ACRES']

fig, ax = plt.subplots(figsize=(12, 10))
crime_merged_map.plot(column='total_crime', cmap='OrRd', linewidth=1.0, ax=ax, edgecolor='0.5', legend=True)
plt.title('Crime Count by Census Tract')
plt.show()
plt.savefig('crime_tract_map.png')

house_merged_map = tract_map.merge(tract_house, left_on='TRACT', right_on='census_tract', how='left')

# Data Visualization of Median House Price by Census Tract on Map
fig, ax = plt.subplots(figsize=(12, 10))
house_merged_map.plot(column='price_median', cmap='Blues', linewidth=1.0, ax=ax, edgecolor='0.5', legend=True)
plt.title('Median Property Value by Census Tract')
plt.show()
plt.savefig('house_tract_map.png')


#%%

# Load ward shapefile
ward_map = gpd.read_file('/Users/chengyuhan/Downloads/Wards_from_2022/Wards_from_2022.shp') 

# Clean up 'WARD' variable if necessary and convert to appropriate type
ward_map['WARD'] = ward_map['WARD'].astype(int)

# Assuming you have a dataframe with crime data and median house prices for wards, named 'ward_crime' and 'ward_house' respectively
# Merge shapefile data with crime data for wards
crime_merged_ward_map = ward_map.merge(ward_crime, left_on='WARD', right_on='ward', how='left')

# Data Visualization of Crime Counts by Ward on Map
fig, ax = plt.subplots(figsize=(12, 10))
crime_merged_ward_map.plot(column='total_crime', cmap='OrRd', linewidth=1.0, ax=ax, edgecolor='0.5', legend=True)
# Add ward labels at centroids
for idx, row in ward_map.iterrows():
    centroid = row['geometry'].centroid
    plt.text(centroid.x, centroid.y, str(row['WARD']), fontsize=15, ha='center', color='black')

plt.title('Crime Count by Ward')
plt.savefig('crime_ward_map.png')
plt.show()

# Merge shapefile data with housing data for wards
house_merged_ward_map = ward_map.merge(ward_house, left_on='WARD', right_on='ward', how='left')

# Data Visualization of Median House Price by Ward on Map
fig, ax = plt.subplots(figsize=(12, 10))
house_merged_ward_map.plot(column='price_median', cmap='Blues', linewidth=1.0, ax=ax, edgecolor='0.5', legend=True)
# Add ward labels at centroids
for idx, row in house_merged_ward_map.iterrows():
#     centroid = row['geometry'].centroid
#     plt.text(centroid.x, centroid.y, str(row['WARD']), fontsize=15, ha='center', color='black')

    centroid = row['geometry'].centroid
    price_median = row['price_median']
    
    # Set color based on the median price
    text_color = 'white' if price_median > 400000 else 'black'
    
    # Add the ward label
    plt.text(centroid.x, centroid.y, str(row['WARD']), fontsize=15, ha='center', color=text_color)

plt.title('Median Property Value by Ward')
plt.savefig('house_ward_map.png')
plt.show()

#%%

# Q2.1 Does different offense types of crime have distinct effect on property value?

offenses = ['offense_arson', 'offense_assault w/dangerous weapon', 'offense_burglary',
       'offense_homicide', 'offense_motor vehicle theft', 'offense_robbery',
       'offense_sex abuse', 'offense_theft f/auto', 'offense_theft/other']

correlation_matrix = cp_data_cleaned[['price'] + offenses].corr(method = 'spearman')
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Price vs Crime Counts')
plt.show()

for offense in offenses:
    corr_coef, p_value = spearmanr(cp_data_cleaned[offense], cp_data_cleaned['price'])
    print(f'{offense}: Spearman Correlation = {corr_coef:.4f}, p-value = {p_value:.4e}')


# The strongest relationships are observed with offenses like assault with a dangerous weapon (-0.3070) and homicide (-0.2293), 
# which suggests a moderate negative association with housing prices. 
# However, most other crime types, such as arson and theft/other, show very weak correlations, 
# indicating that these crimes have only a minor impact on housing prices.
# With all correlation coefficients statistically significant, these findings highlight the potential but varying influence of different crime types on property values.

#%%
# Q2.2 Does methods of committing crime have distinct effect on property value

methods = ['method_gun', 'method_knife', 'method_others']

correlation_matrix = cp_data_cleaned[['price'] + methods].corr(method = 'spearman')
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Price vs Crime Counts')
plt.show()


for method in methods:
    corr_coef, p_value = spearmanr(cp_data_cleaned[method], cp_data_cleaned['price'])
    print(f'{method}: Spearman Correlation = {corr_coef:.4f}, p-value = {p_value:.4e}')

# Method involving guns and knives show moderate negative correlations with the target variable, suggesting that these crime methods have a potentially meaningful inverse impact on property value.
# Method involving others has a very weak positive correlation with the target variable, suggesting that its impact is minimal.
# All the p-values are very small, indicating that these correlations are statistically significant.


#%%

# Q2.3 Does the timing of crime have distinct effect on property value?
shifts = ['shift_day', 'shift_midnight', 'shift_evening']

correlation_matrix = cp_data_cleaned[['price', 'shift_day', 'shift_midnight', 'shift_evening']].corr(method='spearman')
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Price vs Crime Counts')
plt.show()



# List of variables you want to plot
variables = ['price', 'shift_day', 'shift_midnight', 'shift_evening']

# Create pairplot
sns.pairplot(cp_data_cleaned[variables], diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Pairplot of Price and Crime Shifts', y=1.02)
plt.show()

from scipy.stats import spearmanr

for shift in shifts:
    corr_coef, p_value = spearmanr(cp_data_cleaned[shift], cp_data_cleaned['price'])
    print(f'{shift}: Spearman Correlation = {corr_coef:.4f}, p-value = {p_value:.4e}')

# Even though the p-value of correlation coefficients involving different shifts of crime occurrence ('day', 'midnight', and 'evening') are 
# extremely low, the actual strength of these relationsips is weak (close to zero).
# This means that shift variables have only minimal impact on the target variable.

#%% [markdown]

# - The type of offense and the method of committing crime appear to have 
# a more significant influence on property values compared to time of occurence
# - Crimes that are violent in nature seems to have a substantial negative effect on property values.


#%%

#################
### Modelling ###
#################

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv=5, scoring='neg_root_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10), title='Learning Curve', cat_features=None):

    plt.figure(figsize=(10, 6))
    
    fit_params = {'categorical_feature': cat_features} if cat_features else {}

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1,
        fit_params=fit_params
    )

    # Compute mean and standard deviation
    train_mean = -train_scores.mean(axis=1)
    test_mean = -test_scores.mean(axis=1)

    # Plot learning curve
    plt.subplots(figsize=(10, 8))
    plt.plot(train_sizes, train_mean, label="Train RMSE", marker='o')
    plt.plot(train_sizes, test_mean, label="Validation RMSE", marker='o')

    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

#%%


### Total Crime Counts ####

# Control for the physical features (`bathrm`, `rooms`, `bedrm`, `fireplaces`),
# location (`ward`), time (`year`), socioeconomic factor (`median_gross_income`)
controls = ['bathrm', 'rooms', 'bedrm', 'fireplaces', 'year', 'ward', 'median_gross_income']
total = ['total_crime_count']

features = controls + total

X = cp_cleaned[features]
y = cp_cleaned['price']

# Replace spaces with underscores in column names for consistency
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
num_features = [feature for feature in features if feature != 'ward']
cat_features = ['ward']
X[cat_features] = X[cat_features].astype('category')

# Split into training, validation, test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  

# Standardize numerical features
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

#Define initial model parameters
initial_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': 42
}

# Initialize the LGBMRegressor with the specified parameters
model = LGBMRegressor(**initial_params)

# Fit the model with early stopping
model.fit(
    X_train, y_train,
    eval_metric='rmse',
    categorical_feature=cat_features
)

# Training RMSE and R2
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration_)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Initial Training RMSE: {train_rmse:.4f}')
print(f'Initial Training R²: {train_r2:.4f}')

# Test RMSE and R2
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)
print(f'Initial Test RMSE: {test_rmse:.4f}')
print(f'Initial Test R²: {test_r2:.4f}')

#%%
# Cross Validation Score for Initial Model

lgbm_cv = LGBMRegressor(**initial_params)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_validate(
    lgbm_cv,
    X_train,
    y_train,
    cv=kf,
    scoring=['neg_root_mean_squared_error', 'r2'],
    n_jobs=1,
    verbose=0,
    fit_params={'categorical_feature': cat_features}
)

cv_rmse = -cv_scores['test_neg_root_mean_squared_error']
cv_r2 = cv_scores['test_r2']

# Cross Validation RMSE and R2 
print(f'Cross-Validation RMSE Scores: {cv_rmse}')
print(f'Average Cross-Validation RMSE: {cv_rmse.mean():.4f}')

print(f'Cross-Validation R² Scores: {cv_r2}')
print(f'Average Cross-Validation R²: {cv_r2.mean():.4f}')

#%% 
######### Hyperparameter Tuning ################# 

param_grid = {
    'learning_rate': [0.03, 0.05, 0.07],
    'num_leaves': [20, 27, 31],
    'max_depth': [7, 10, 15],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.3]
}

lgbm = LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    verbosity=-1,
    random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=kf,
    verbose=1,
    n_jobs=1
)

grid_search.fit(X_train, y_train,categorical_feature=cat_features)

# Extract the best parameters and best CV RMSE
best_params = grid_search.best_params_
best_rmse = -grid_search.best_score_
print('Best Parameters found by GridSearchCV:', best_params)
print(f'Best CV RMSE: {best_rmse:.4f}')


# Fitting 5 folds for each of 243 candidates, totalling 1215 fits
# Best Parameters found by GridSearchCV: {'learning_rate': 0.07, 'max_depth': 15, 'num_leaves': 31, 'reg_alpha': 0, 'reg_lambda': 0.1}
# Best CV RMSE: 134926.1966
#%%

best_params = {'learning_rate': 0.07, 'max_depth': 15, 'num_leaves': 31, 'reg_alpha': 0, 'reg_lambda': 0.1}

model_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    **best_params
}

lgbm_cv = LGBMRegressor(**model_params)

cv_scores_best = cross_validate(
    lgbm_cv, 
    X_train,
    y_train,
    cv=kf,
    scoring=['neg_root_mean_squared_error', 'r2'],
    n_jobs=1,
    verbose=0,
    fit_params={'categorical_feature': cat_features}
)


final_model = LGBMRegressor(**model_params)

# Plot Learning Curve for Final Model
# plot_learning_curve(
#     estimator=final_model,
#     X=X_train,
#     y=y_train,
#     cv=5,
#     scoring='neg_root_mean_squared_error',
#     title='Learning Curve - Final Model',
#     cat_features=cat_features
# )

final_model.fit(
    X_train, y_train,
    eval_metric='rmse',
    categorical_feature=cat_features
)


####### Final Model Evaluation ############

# Calculate Train Data RMSE and R2
y_train_pred = final_model.predict(X_train, num_iteration=final_model.best_iteration_)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate Test Data RMSE and R2
y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration_)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)

# Calculate Cross Validation RMSE and R2 
cv_rmse_best = -cv_scores_best['test_neg_root_mean_squared_error']
cv_r2_best = cv_scores_best['test_r2']

plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (Total Crime Counts)')
plt.show()


# Print out outputs
print(f'Final Model Train RMSE: {train_rmse:.4f}')
print(f'Final Model Train R²: {train_r2:.4f}\n')

print(f'Final Model Test RMSE: {test_rmse:.4f}')
print(f'Final Model Test R²: {test_r2:.4f}\n')

print(f'Average Cross-Validation RMSE (Best Params): {cv_rmse_best.mean():.4f}')
print(f'Average Cross-Validation R² (Best Params): {cv_r2_best.mean():.4f}')

#%%

# Extract feature importances
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
})
# Sort features by importance in descending order
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print("Feature Importances:")
print(feature_importances)

# Plot feature importances by gain
lgb.plot_importance(final_model, importance_type="gain", figsize=(7,6), title="LightGBM Feature Importance (Gain)")
plt.show()

# Plot feature importances by split
lgb.plot_importance(final_model, importance_type="split", figsize=(7,6), title="LightGBM Feature Importance (Split)")
plt.show()



#%% [markdown]

#### All Featrues ###

offenses = ['violent_crime_count', 'property_crime_count']

features = controls+offenses+methods+shifts

X = cp_cleaned[features]
y = cp_cleaned['price']

# Clean up column name
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
num_features = [feature for feature in features if feature != 'ward']
cat_features = ['ward']
X[cat_features] = X[cat_features].astype('category')

# Split into training, validation, test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  

# Standardize numerical features
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])


initial_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'random_state': 42
}

initial_model = LGBMRegressor(**initial_params)

initial_model.fit(
    X_train, y_train,
    categorical_feature=cat_features,
    eval_metric='rmse'
)

y_train_pred_initial = initial_model.predict(X_train, num_iteration=initial_model.best_iteration_)
train_rmse_initial = mean_squared_error(y_train, y_train_pred_initial, squared=False)
train_r2_initial = r2_score(y_train, y_train_pred_initial)

print(f'Initial Model Train RMSE: {train_rmse:.4f}')
print(f'Initial Model Train R²: {train_r2:.4f}\n')

#%%
# Cross Validation Score for Initial Model

lgbm_cv = LGBMRegressor(**initial_params)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_validate(
    lgbm_cv,
    X_train,
    y_train,
    cv=kf,
    scoring=['neg_root_mean_squared_error', 'r2'],
    n_jobs=1,
    verbose=0,
    fit_params={'categorical_feature': cat_features}
)

cv_rmse = -cv_scores['test_neg_root_mean_squared_error']
cv_r2 = cv_scores['test_r2']

# Cross Validation RMSE and R2 
print(f'Cross-Validation RMSE Scores: {cv_rmse}')
print(f'Average Cross-Validation RMSE: {cv_rmse.mean():.4f}')

print(f'Cross-Validation R² Scores: {cv_r2}')
print(f'Average Cross-Validation R²: {cv_r2.mean():.4f}')



#%%
#### All Featrues ###

from sklearn.preprocessing import StandardScaler

offenses = ['violent_crime_count', 'property_crime_count']

features = controls + offenses + methods + shifts

# Assuming cp_data_cleaned is your DataFrame and controls, offenses, methods, shifts are your feature lists
X = cp_data_cleaned[features]
y = cp_data_cleaned['price']


# Clean up column names
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
num_features = [feature for feature in features if feature != 'ward']
cat_features = ['ward']
X[cat_features] = X[cat_features].astype('category')



# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# Standardize numerical features
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

initial_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'dart',
    'random_state': 42
}


initial_model = LGBMRegressor(**initial_params, categorical_feature = X_train.columns.get_loc('ward'))

initial_model.fit(
    X_train, y_train,
    eval_metric='rmse',
    categorical_feature=cat_features
)

y_train_pred_initial = initial_model.predict(X_train, num_iteration=initial_model.best_iteration_)
train_rmse_initial = mean_squared_error(y_train, y_train_pred_initial, squared=False)
train_r2_initial = r2_score(y_train, y_train_pred_initial)

y_test_pred_initial = initial_model.predict(X_test, num_iteration=initial_model.best_iteration_)
test_rmse_initial = mean_squared_error(y_test, y_test_pred_initial, squared=False)
test_r2_initial = r2_score(y_test, y_test_pred_initial)

print(f'Initial Model Train RMSE: {train_rmse:.4f}')
print(f'Initial Model Train R²: {train_r2:.4f}\n')

print(f"Initial Model Test RMSE: {test_rmse_initial:.4f}")
print(f"Initial Model Test R²: {test_r2_initial:.4f}")

#%%
# Cross Validation Score for Initial Model

lgbm_cv = LGBMRegressor(**initial_params)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_validate(
    lgbm_cv,
    X_train,
    y_train,
    cv=kf,
    scoring=['neg_root_mean_squared_error', 'r2'],
    n_jobs=1,
    verbose=0,
    fit_params={'categorical_feature': cat_features}
)

cv_rmse = -cv_scores['test_neg_root_mean_squared_error']
cv_r2 = cv_scores['test_r2']

# Cross Validation RMSE and R2 
print(f'Cross-Validation RMSE Scores: {cv_rmse}')
print(f'Average Cross-Validation RMSE: {cv_rmse.mean():.4f}')

print(f'Cross-Validation R² Scores: {cv_r2}')
print(f'Average Cross-Validation R²: {cv_r2.mean():.4f}')

#%% 
# Hyperparameter Tuning 

param_grid = {
    'learning_rate': [0.01, 0.05, 0.07, 0.1],
    'num_leaves': [20, 31, 62],
    'max_depth': [7, 10, 15],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

lgbm = LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    categorical_feature=X_train.columns.get_loc('ward'),
    verbosity=-1,
    random_state=42

)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error', 
    cv=kf,
    verbose=1,
    n_jobs=1
)

grid_search.fit(X_train, y_train, categorical_feature=cat_features)

# Extract the best parameters and best CV RMSE
best_params = grid_search.best_params_
best_rmse = -grid_search.best_score_
print('Best Parameters found by GridSearchCV:', best_params)
print(f'Best CV RMSE: {best_rmse:.4f}')

# Best Parameters found by GridSearchCV: {'learning_rate': 0.1, 'max_depth': 15, 'num_leaves': 62, 'reg_alpha': 0.5, 'reg_lambda': 0.1}
# Best CV RMSE: 132909.6492
#%%
best_params =  {'learning_rate': 0.1, 'max_depth': 15, 'num_leaves': 62, 'reg_alpha': 0.5, 'reg_lambda': 0.1}

model_params ={
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    **best_params}


# Initialize cross validation
lgbm_cv = LGBMRegressor(**model_params)

cv_scores_best = cross_validate(
    lgbm_cv, 
    X_train,
    y_train,
    cv=kf,
    scoring=['neg_root_mean_squared_error', 'r2'],
    n_jobs=1,
    verbose=0,
    fit_params={'categorical_feature': cat_features}
)

# Initialize the final model with the best parameters
final_model = LGBMRegressor(**model_params)

# #Plot Learning Curve for Final Model
# plot_learning_curve(
#     estimator=final_model,
#     X=X_train,
#     y=y_train,
#     cv=5,
#     scoring='neg_root_mean_squared_error',
#     title='Learning Curve - Final Model',
#     cat_features=cat_features
# )


# Fit model on training data set
final_model.fit(
    X_train, y_train,
    eval_metric='rmse',
    categorical_feature = X_train.columns.get_loc('ward')
)

# Evaluate Final Model
# Calculate Train Data RMSE and R2
y_train_pred = final_model.predict(X_train, num_iteration=final_model.best_iteration_)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate Test Data RMSE and R2
y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration_)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)

# Calculate Cross Validation RMSE and R2 
cv_rmse_best = -cv_scores_best['test_neg_root_mean_squared_error']
cv_r2_best = cv_scores_best['test_r2']

print(f'Final Model Train RMSE: {train_rmse:.4f}')
print(f'Final Model Train R²: {train_r2:.4f}\n')

print(f'Final Model Test RMSE: {test_rmse:.4f}')
print(f'Final Model Test R²: {test_r2:.4f}\n')

print(f'Average Cross-Validation RMSE (Best Params): {cv_rmse_best.mean():.4f}')
print(f'Average Cross-Validation R² (Best Params): {cv_r2_best.mean():.4f}')

plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (All Features)')
plt.show()



#%% Final Feature Importance Analysis

# Extract feature importances from the final model
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
})

# Sort features by importance in descending order
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print("Final Feature Importances:")
print(feature_importances)


lgb.plot_importance(final_model, importance_type="gain", figsize=(7,6), title="LightGBM Feature Importance (Gain)")
plt.show()

lgb.plot_importance(final_model, importance_type="split", figsize=(7,6), title="LightGBM Feature Importance (Split)")
plt.show()


#%%

cp_data_cleaned.ward.unique()
# %%
