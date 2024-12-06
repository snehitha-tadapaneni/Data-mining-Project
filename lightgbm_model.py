# %%

#%% Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor



#%%
######################################
############# LightGBM ###############
######################################



# Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor



#%% Load Data
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

# %%
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import lightgbm as lgb
from lightgbm import LGBMRegressor

#######################
###### LightGBM ######
######################

#%%
#####################################
#### Without Crime Statistics #######
#####################################

house = ['bathrm', 'rooms', 'bedrm', 'fireplaces', 'ward', 'median_gross_income',
       'year']

X = cp_data_cleaned[house]
y = cp_data_cleaned['price']


# Replace spaces with underscores in column names for consistency
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
cat_features = ['ward']
X[cat_features] = X[cat_features].astype('category')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define initial model parameters, including a high n_estimators
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
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)


# ### Model Evaluation ###

# Predict on Training Data
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration_)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Initial Training RMSE: {train_rmse:.4f}')
print(f'Initial Training R²: {train_r2:.4f}')

# Predict on Validation Data
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
val_r2 = r2_score(y_val, y_val_pred)
print(f'Initial Validation RMSE: {val_rmse:.4f}')
print(f'Initial Validation R²: {val_r2:.4f}')

#%%

#####################
### ALL Features ####
#####################


# ### Initial Stage ###

X = cp_data_cleaned.drop(['price'], axis=1)
y = cp_data_cleaned['price']


# Replace spaces with underscores in column names for consistency
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
cat_features = ['ward']
X[cat_features] = X[cat_features].astype('category')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define initial model parameters, including a high n_estimators
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
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)


# ### Model Evaluation ###

# Predict on Training Data
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration_)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Initial Training RMSE: {train_rmse:.4f}')
print(f'Initial Training R²: {train_r2:.4f}')

# Predict on Validation Data
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
val_r2 = r2_score(y_val, y_val_pred)
print(f'Initial Validation RMSE: {val_rmse:.4f}')
print(f'Initial Validation R²: {val_r2:.4f}')

# %%
# ### Understanding Feature Importance ###

# Extract feature importances
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
})

# Sort features by importance in descending order
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print("Feature Importances:")
print(feature_importances)


# %%
# ### Hyperparameter Tuning with GridSearchCV ###

# Initialize the LGBMRegressor
lgbm = LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    verbosity=-1,
    random_state=42,
    n_estimators=1000
)

# Define the parameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 62],
    'max_depth': [7, 10, 15],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Initialize cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',  # Negative RMSE for minimization
    cv=kf,
    verbose=2,
    n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)

# Best parameters and corresponding score
best_params = grid_search.best_params_
best_rmse = -grid_search.best_score_
print('Best Parameters found by GridSearchCV:', best_params)
print(f'Best CV RMSE: {best_rmse:.4f}')

# Best Parameters found by GridSearchCV: {'learning_rate': 0.05, 'max_depth': 15, 'num_leaves': 31, 'reg_alpha': 0.5, 'reg_lambda': 0}
# Best CV RMSE: 133073.8111

# %%
# ### Final Model with Best Parameters ###

# Update the initial parameters with the best parameters found
final_params = initial_params.copy()
final_params.update(best_params)

print(final_params)

# Initialize the final model with updated parameters
final_model = LGBMRegressor(**final_params)

# Fit the final model with early stopping
final_model.fit(
    X_train, y_train,
    # eval_set=[(X_val, y_val)],
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=['training', 'validation'],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)

# Predict on Training Data
y_train_pred_final = final_model.predict(X_train, num_iteration=final_model.best_iteration_)
final_train_rmse = mean_squared_error(y_train, y_train_pred_final, squared=False)
final_train_r2 = r2_score(y_train, y_train_pred_final)
print(f'Final Training RMSE: {final_train_rmse:.4f}')
print(f'Final Training R²: {final_train_r2:.4f}')

# Predict on Validation Data
y_val_pred_final = final_model.predict(X_val, num_iteration=final_model.best_iteration_)
final_val_rmse = mean_squared_error(y_val, y_val_pred_final, squared=False)
final_val_r2 = r2_score(y_val, y_val_pred_final)
print(f'Final Validation RMSE: {final_val_rmse:.4f}')
print(f'Final Validation R²: {final_val_r2:.4f}')

#%%
# ### Cross-Validation ###


# Initialize the LGBMRegressor with initial parameters
lgbm_cv = LGBMRegressor(**final_params)

# Perform cross-validation using cross_val_score
cv_scores = cross_val_score(
    lgbm_cv,
    X,
    y,
    cv=kf,
    scoring='neg_root_mean_squared_error',  # Negative RMSE because higher return values are better in scikit-learn
    n_jobs=-1,
    verbose=10
)

# Convert negative RMSE to positive
cv_rmse = -cv_scores
print(f'Cross-Validation RMSE Scores: {cv_rmse}')
print(f'Average Cross-Validation RMSE: {cv_rmse.mean():.4f}')
print(f'Standard Deviation of CV RMSE: {cv_rmse.std():.4f}')

# %%
# ### Final Feature Importance Analysis ###

# Extract feature importances from the final model
final_feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
})

# Sort features by importance in descending order
final_feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print("Final Feature Importances:")
print(final_feature_importances)

#%%
####################
##### Offense ######
####################


offenses = ['bathrm', 'rooms', 'bedrm', 'fireplaces', 'ward','census_tract',
       'year', 'median_gross_income', 'offense_arson',
       'offense_assault w/dangerous weapon', 'offense_burglary',
       'offense_homicide', 'offense_motor vehicle theft', 'offense_robbery',
       'offense_sex abuse', 'offense_theft f/auto', 'offense_theft/other']


# ### Initial Stage ###

X = cp_data_cleaned[offenses]
y = cp_data_cleaned['price']


# Replace spaces with underscores in column names for consistency
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
cat_features = ['ward']
X[cat_features] = X[cat_features].astype('category')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define initial model parameters, including a high n_estimators
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
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)


# ### Model Evaluation ###

# Predict on Training Data
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration_)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Initial Training RMSE: {train_rmse:.4f}')
print(f'Initial Training R²: {train_r2:.4f}')

# Predict on Validation Data
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
val_r2 = r2_score(y_val, y_val_pred)
print(f'Initial Validation RMSE: {val_rmse:.4f}')
print(f'Initial Validation R²: {val_r2:.4f}')

# %%
# ### Understanding Feature Importance ###

# Extract feature importances
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
})

# Sort features by importance in descending order
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print("Feature Importances:")
print(feature_importances)


#%%

# ### Cross-Validation ###

# Initialize K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the LGBMRegressor with initial parameters
lgbm_cv = LGBMRegressor(**initial_params)

# Perform cross-validation using cross_val_score
cv_scores = cross_val_score(
    lgbm_cv,
    X,
    y,
    cv=kf,
    scoring='neg_root_mean_squared_error',  # Negative RMSE because higher return values are better in scikit-learn
    n_jobs=-1,
    verbose=1
)

# Convert negative RMSE to positive
cv_rmse = -cv_scores
print(f'Cross-Validation RMSE Scores: {cv_rmse}')
print(f'Average Cross-Validation RMSE: {cv_rmse.mean():.4f}')
print(f'Standard Deviation of CV RMSE: {cv_rmse.std():.4f}')



# ### Hyperparameter Tuning with GridSearchCV ###

# Initialize the LGBMRegressor
lgbm = LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    verbosity=-1,
    random_state=42,
    n_estimators=1000
)

# Define the parameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 62],
    'max_depth': [7, 10, 15],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Initialize cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',  # Negative RMSE for minimization
    cv=kf,
    verbose=2,
    n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(
    X_train, y_train,
    # eval_set=[(X_val, y_val)],
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)

# Best parameters and corresponding score
best_params = grid_search.best_params_
best_rmse = -grid_search.best_score_
print('Best Parameters found by GridSearchCV:', best_params)
print(f'Best CV RMSE: {best_rmse:.4f}')

# Best Parameters found by GridSearchCV: {'learning_rate': 0.05, 'max_depth': 15, 'num_leaves': 31, 'reg_alpha': 0.5, 'reg_lambda': 0}
# Best CV RMSE: 133073.8111

# %%
# ### Final Model with Best Parameters ###

# Update the initial parameters with the best parameters found
final_params = initial_params.copy()
final_params.update(best_params)

final_params

#%%

# Initialize the final model with updated parameters
final_model = LGBMRegressor(**final_params)

# Fit the final model with early stopping
final_model.fit(
    X_train, y_train,
    # eval_set=[(X_val, y_val)],
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=['training', 'validation'],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)

# Predict on Training Data
y_train_pred_final = final_model.predict(X_train, num_iteration=final_model.best_iteration_)
final_train_rmse = mean_squared_error(y_train, y_train_pred_final, squared=False)
final_train_r2 = r2_score(y_train, y_train_pred_final)
print(f'Final Training RMSE: {final_train_rmse:.4f}')
print(f'Final Training R²: {final_train_r2:.4f}')

# Predict on Validation Data
y_val_pred_final = final_model.predict(X_val, num_iteration=final_model.best_iteration_)
final_val_rmse = mean_squared_error(y_val, y_val_pred_final, squared=False)
final_val_r2 = r2_score(y_val, y_val_pred_final)
print(f'Final Validation RMSE: {final_val_rmse:.4f}')
print(f'Final Validation R²: {final_val_r2:.4f}')

#%%
# ### Cross-Validation ###

# Initialize K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the LGBMRegressor with initial parameters
lgbm_cv = LGBMRegressor(**initial_params)

# Perform cross-validation using cross_val_score
cv_scores = cross_val_score(
    lgbm_cv,
    X,
    y,
    cv=kf,
    scoring='neg_root_mean_squared_error',  # Negative RMSE because higher return values are better in scikit-learn
    n_jobs=-1,
    verbose=10
)

# Convert negative RMSE to positive
cv_rmse = -cv_scores
print(f'Cross-Validation RMSE Scores: {cv_rmse}')
print(f'Average Cross-Validation RMSE: {cv_rmse.mean():.4f}')
print(f'Standard Deviation of CV RMSE: {cv_rmse.std():.4f}')

# %%
# ### Final Feature Importance Analysis ###

# Extract feature importances from the final model
final_feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
})

# Sort features by importance in descending order
final_feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print("Final Feature Importances:")
print(final_feature_importances)

# %%


################
#### Mehtod ####
################


methods = ['bathrm', 'rooms', 'bedrm', 'fireplaces', 'ward', 'census_tract',
       'year', 'median_gross_income', 'method_gun', 'method_knife', 'method_others']


# ### Initial Stage ###

X = cp_data_cleaned[methods]
y = cp_data_cleaned['price']


# Replace spaces with underscores in column names for consistency
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
cat_features = ['ward']
X[cat_features] = X[cat_features].astype('category')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define initial model parameters, including a high n_estimators
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
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)


# ### Model Evaluation ###

# Predict on Training Data
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration_)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Initial Training RMSE: {train_rmse:.4f}')
print(f'Initial Training R²: {train_r2:.4f}')

# Predict on Validation Data
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
val_r2 = r2_score(y_val, y_val_pred)
print(f'Initial Validation RMSE: {val_rmse:.4f}')
print(f'Initial Validation R²: {val_r2:.4f}')

# %%
# ### Understanding Feature Importance ###

# Extract feature importances
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
})

# Sort features by importance in descending order
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print("Feature Importances:")
print(feature_importances)


#%%


#############
### Shift ###
#############

methods = ['bathrm', 'rooms', 'bedrm', 'fireplaces', 'ward','census_tract',
       'year', 'median_gross_income', 'shift_day', 'shift_evening', 'shift_midnight']


# ### Initial Stage ###

X = cp_data_cleaned[methods]
y = cp_data_cleaned['price']


# Replace spaces with underscores in column names for consistency
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
cat_features = ['ward', 'year']
X[cat_features] = X[cat_features].astype('category')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define initial model parameters, including a high n_estimators
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
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)


# ### Model Evaluation ###

# Predict on Training Data
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration_)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Initial Training RMSE: {train_rmse:.4f}')
print(f'Initial Training R²: {train_r2:.4f}')

# Predict on Validation Data
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
val_r2 = r2_score(y_val, y_val_pred)
print(f'Initial Validation RMSE: {val_rmse:.4f}')
print(f'Initial Validation R²: {val_r2:.4f}')

#%%


####################
### Total Crimes ###
####################

cp_data_cleaned['total_crime_count'] = cp_data_cleaned[
    ['offense_assault w/dangerous weapon', 'offense_homicide', 
     'offense_robbery', 'offense_sex abuse', 'offense_arson',
    'offense_burglary', 'offense_motor vehicle theft', 'offense_theft f/auto', 
    'offense_theft/other']].sum(axis=1)


total = ['bathrm', 'rooms', 'bedrm', 'fireplaces', 'ward','census_tract', 'year','total_crime_count']

X = cp_data_cleaned[total]
y = cp_data_cleaned['price']


# Replace spaces with underscores in column names for consistency
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
cat_features = ['ward']
X[cat_features] = X[cat_features].astype('category')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define initial model parameters, including a high n_estimators
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
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)


# ### Model Evaluation ###

# Predict on Training Data
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration_)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Initial Training RMSE: {train_rmse:.4f}')
print(f'Initial Training R²: {train_r2:.4f}')

# Predict on Validation Data
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
val_r2 = r2_score(y_val, y_val_pred)
print(f'Initial Validation RMSE: {val_rmse:.4f}')
print(f'Initial Validation R²: {val_r2:.4f}')


# %%

###########################
# train, validation, test #
###########################


#####################################
#### Without Crime Statistics #######
#####################################

house = ['bathrm', 'rooms', 'bedrm', 'fireplaces','ward',
       'year']

X = cp_data_cleaned[house]
y = cp_data_cleaned['price']


# Replace spaces with underscores in column names for consistency
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
cat_features = ['ward']
X[cat_features] = X[cat_features].astype('category')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define initial model parameters, including a high n_estimators
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
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)


# ### Model Evaluation ###

# Predict on Training Data
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration_)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Initial Training RMSE: {train_rmse:.4f}')
print(f'Initial Training R²: {train_r2:.4f}')

# Predict on Validation Data
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
val_r2 = r2_score(y_val, y_val_pred)
print(f'Initial Validation RMSE: {val_rmse:.4f}')
print(f'Initial Validation R²: {val_r2:.4f}')

#%%
cp_data.columns

['bathrm', 'rooms', 'bedrm', 'price', 'fireplaces', 'census_tract',
       'ward', 'year', 'median_gross_income', 'offense_arson',
       'offense_assault w/dangerous weapon', 'offense_burglary',
       'offense_homicide', 'offense_motor vehicle theft', 'offense_robbery',
       'offense_sex abuse', 'offense_theft f/auto', 'offense_theft/other',
       'method_gun', 'method_knife', 'method_others', 'shift_day',
       'shift_evening', 'shift_midnight']
##### ALL FEATURES ######

#%% Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt

#%% Split Data into Train, Validation, and Test
X = cp_data_cleaned.drop('price', axis=1)
y = cp_data_cleaned['price']

# Replace spaces with underscores in column names for consistency
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features
cat_features = ['ward']

# Convert categorical features to 'category' dtype
X[cat_features] = X[cat_features].astype('category')

# Split into training, validation, test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)  

#%% Hyperparameter Tuning with GridSearchCV on Training Data
# Define the parameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 62],
    'max_depth': [7, 10, 15],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Initialize the LGBMRegressor
lgbm = LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    verbosity=-1,
    random_state=42,
    n_estimators=1000
)


# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',  # Negative RMSE for minimization
    cv=5,
    verbose=2,
    n_jobs=-1
)

# Fit GridSearchCV on the Training Set
grid_search.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)

# Retrieve the best parameters and corresponding score
best_params = grid_search.best_params_
best_rmse = -grid_search.best_score_
print('Best Parameters found by GridSearchCV:', best_params)
print(f'Best CV RMSE: {best_rmse:.4f}')

# Best Parameters found by GridSearchCV: {'learning_rate': 0.05, 'max_depth': 15, 'num_leaves': 31, 'reg_alpha': 0.5, 'reg_lambda': 0}
# Best CV RMSE: 134526.5459
#%% 

best_params = {'learning_rate': 0.05, 'max_depth': 15, 'num_leaves': 31, 'reg_alpha': 0.5, 'reg_lambda': 0}

# Initialize the final model with the best parameters
final_model = LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    verbosity=-1,
    random_state=42,
    n_estimators=1000,
    **best_params
)

# Fit the final model
final_model.fit(
    X_train_val, y_train_val,
    eval_set=[(X_train_val, y_train_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)

#%% 

# Evaluate Final Model on Test Set
# Predict on the Test Set
y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration_)

# Calculate RMSE and R²
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Test RMSE: {test_rmse:.4f}')
print(f'Test R²: {test_r2:.4f}')

#%% Final Feature Importance Analysis
# Extract feature importances from the final model
final_feature_importances = pd.DataFrame({
    'feature': X_train_val.columns,
    'importance': final_model.feature_importances_
})

# Sort features by importance in descending order
final_feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print("Final Feature Importances:")
print(final_feature_importances)

# %%
cp_data_cleaned['total_crime'] = cp_data_cleaned[[
    'offense_assault w/dangerous weapon', 'offense_homicide', 
    'offense_robbery', 'offense_sex abuse', 'offense_arson',
    'offense_burglary', 'offense_motor vehicle theft', 'offense_theft f/auto', 
    'offense_theft/other']].sum(axis=1)
cp_data_cleaned['violent_crime_count'] = cp_data_cleaned[['offense_assault w/dangerous weapon', 'offense_homicide', 'offense_robbery', 'offense_sex abuse']].sum(axis=1)
cp_data_cleaned['property_crime_count'] = cp_data_cleaned[['offense_arson', 'offense_burglary', 'offense_motor vehicle theft', 'offense_theft f/auto', 'offense_theft/other']].sum(axis=1)

# Compute ratios
cp_data_cleaned['violent_crime_ratio'] = cp_data_cleaned['violent_crime_count']/cp_data_cleaned['total_crime']
cp_data_cleaned['property_crime_ratio'] = cp_data_cleaned['property_crime_count']/cp_data_cleaned['total_crime']


cp_data_cleaned['gun_crime_ratio'] = cp_data_cleaned['method_gun']/cp_data_cleaned['total_crime']
cp_data_cleaned['knife_crime_ratio'] = cp_data_cleaned['method_knife']/cp_data_cleaned['total_crime']
cp_data_cleaned['other_methods_crime_ratio'] = cp_data_cleaned['method_others']/cp_data_cleaned['total_crime']

cp_data_cleaned['day_crime_ratio'] = cp_data_cleaned['shift_day'] / cp_data_cleaned['total_crime_count']
cp_data_cleaned['evening_crime_ratio'] = cp_data_cleaned['shift_evening'] / cp_data_cleaned['total_crime_count']
cp_data_cleaned['midnight_crime_ratio'] = cp_data_cleaned['shift_midnight'] / cp_data_cleaned['total_crime_count']

# #%%

# cp_data_cleaned['hotspot'] = np.where(cp_data_cleaned['total_crime_count'] > cp_data_cleaned['total_crime_count'].mean(), True, False)

#%%
cp_data_cleaned.columns

#%% [markdown]
# Q1: Does neighborhood level crime count affect property value?

# EDA

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
# Accroding to scatter plot, we cannot see a clear patterns between total crime counts and price.
# We will visualize the distribution of crimes and property value for each census tarct on the map.
# Compare and contrast if neighborhood with higher occurence of crime is associated with lower property value.

# %%

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
#%%
# Q1.1 Does different offense types of crime have distinct effect on property value?

import geopandas as gpd

map = gpd.read_file('/Users/chengyuhan/Downloads/Census_Tracts_in_2010.shp')


map['TRACT'] = map['TRACT'].str.lstrip('0').astype(int)


# Data Visualization of Crime Counts by Census Tract on Map
crime_merged_map = map.merge(tract_crime, left_on='TRACT', right_on='census_tract', how='left')
crime_merged_map['crime_rate'] = crime_merged_map['total_crime']/ crime_merged_map['P0010001']
crime_merged_map['crime_density'] = crime_merged_map['total_crime']/ crime_merged_map['ACRES']

fig, ax = plt.subplots(figsize=(12, 10))
crime_merged_map.plot(column='total_crime', cmap='OrRd', linewidth=1.0, ax=ax, edgecolor='0.5', legend=True)
plt.title('Crime Count by Census Tract')
plt.show()


house_merged_map = map.merge(tract_house, left_on='TRACT', right_on='census_tract', how='left')

# Data Visualization of Median House Price by Census Tract on Map
fig, ax = plt.subplots(figsize=(12, 10))
house_merged_map.plot(column='price_median', cmap='YlGn', linewidth=1.0, ax=ax, edgecolor='0.5', legend=True)
plt.title('Median Property Value by Census Tract')
plt.show()

#%%

cp_data_cleaned.columns

#%%
# Q1.1 Does different offense types of crime have distinct effect on property value?

# offenses = ['violent_crime_count', 'property_crime_count']

# /or (should we also group offense types into violent vs property)

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

#%%
# Q1.2 Does methods of committing crime have distinct effect on property value

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

# Q1.3 Does the timing of crime have distinct effect on property value?
shifts = ['shift_day', 'shift_midnight', 'shift_evening']

correlation_matrix = cp_data_cleaned[['price', 'shift_day', 'shift_midnight', 'shift_evening']].corr(method='spearman')
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Price vs Crime Counts')
plt.show()


from scipy.stats import spearmanr

for shift in shifts:
    corr_coef, p_value = spearmanr(cp_data_cleaned[shift], cp_data_cleaned['price'])
    print(f'{shift}: Spearman Correlation = {corr_coef:.4f}, p-value = {p_value:.4e}')

# Even though the p-value of correlation coefficients involving different shifts of crime occurrence ('day', 'midnight', and 'evening') are 
# extremely low, the actual strength of these relationsips is week (close to zero).
# This means that shift variables have only minimal impact on the target variable.

#%%
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

X = cp_data_cleaned.drop(['price', 'hotspot'], axis=1)
y = cp_data_cleaned['price']

X.info()
#%%
# Split into training, validation, test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize RFE with Linear Regression
model = LinearRegression()
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X_train, y_train)

# Get selected features
selected_features = X_train.columns[rfe.support_]
print("Selected Features:", selected_features)

# Initialize and train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = lr.predict(X_test)
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))

import statsmodels.api as sm

# Add constant term for intercept
X_train_sm = sm.add_constant(X_train)

# Fit OLS model
model = sm.OLS(y_train, X_train_sm).fit()

# Summary of the model
print(model.summary())

# offense_arson, offense_bulglary, offense_homocide, offense_theft/other

#%%%

###################
#### Modelling ####
###################

cp_data.columns

#%%

###################
#### LightGBM #####
###################

all = ['bathrm', 'rooms', 'bedrm', 'fireplaces',
       'ward', 'year', 'median_gross_income', 'offense_arson',
       'offense_assault w/dangerous weapon', 'offense_burglary',
       'offense_homicide', 'offense_motor vehicle theft', 'offense_robbery',
       'offense_sex abuse', 'offense_theft f/auto', 'offense_theft/other',
       'method_gun', 'method_knife', 'method_others', 'shift_day',
       'shift_evening', 'shift_midnight']

# should we choose either ward or census_tract to control for geographical similarity?

#%% Split Data into Train, Validation, and Test
X = cp_data_cleaned[all]
y = cp_data_cleaned['price']

# Replace spaces with underscores in column names for consistency
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features
cat_features = ['ward']

# Convert categorical features to 'category' dtype
X[cat_features] = X[cat_features].astype('category')

# Split into training, validation, test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)  

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
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)


# ### Model Evaluation ###
# Predict on Training Data
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration_)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Initial Training RMSE: {train_rmse:.4f}')
print(f'Initial Training R²: {train_r2:.4f}')

# Predict on Validation Data
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
val_r2 = r2_score(y_val, y_val_pred)
print(f'Initial Validation RMSE: {val_rmse:.4f}')
print(f'Initial Validation R²: {val_r2:.4f}')

#%% Hyperparameter Tuning with GridSearchCV on Training Data
# Define the parameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 62],
    'max_depth': [7, 10, 15],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Initialize the LGBMRegressor
lgbm = LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    verbosity=-1,
    random_state=42,
    n_estimators=1000
)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',  # Negative RMSE for minimization
    cv=5,
    verbose=2,
    n_jobs=-1
)

# Fit GridSearchCV on the Training Set
grid_search.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)

# Retrieve the best parameters and corresponding score
best_params = grid_search.best_params_
best_rmse = -grid_search.best_score_
print('Best Parameters found by GridSearchCV:', best_params)
print(f'Best CV RMSE: {best_rmse:.4f}')

# Best Parameters found by GridSearchCV: {'learning_rate': 0.05, 'max_depth': 15, 'num_leaves': 62, 'reg_alpha': 0.1, 'reg_lambda': 0}
# Best CV RMSE: 134928.3608
#%% 

best_params = {'learning_rate': 0.05, 'max_depth': 15, 'num_leaves': 62, 'reg_alpha': 0.1, 'reg_lambda': 0}

# Initialize the final model with the best parameters
final_model = LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    verbosity=-1,
    random_state=42,
    n_estimators=1000,
    **best_params
)

# Fit the final model
final_model.fit(
    X_train_val, y_train_val,
    eval_set=[(X_train_val, y_train_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=cat_features
)

#%% 

# Evaluate Final Model on Test Set
# Predict on the Test Set
y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration_)

# Calculate RMSE and R²
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Test RMSE: {test_rmse:.4f}')
print(f'Test R²: {test_r2:.4f}')

#%% Final Feature Importance Analysis
# Extract feature importances from the final model
final_feature_importances = pd.DataFrame({
    'feature': X_train_val.columns,
    'importance': final_model.feature_importances_
})

# Sort features by importance in descending order
final_feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print("Final Feature Importances:")
print(final_feature_importances)


# %%
