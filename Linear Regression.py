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


from scipy import stats

#%% Load Data
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

#%% 

# Resources: https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/ 
# Resources: https://medium.com/@sahin.samia/ml-series-3-regression-assumption-check-ensuring-the-validity-of-linear-regression-models-using-5d876394ae81 


def check_assumptions(model, X_test, y_test, y_pred):

    residuals = y_test - y_pred
    fitted = y_pred
    
    # 1. Residuals vs Fitted Plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=fitted, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.show()
    
    # 2. Normality Q-Q Plot
    qqplot(residuals, line='45', fit=True)
    plt.title('Q-Q Plot of Residuals')
    plt.show()

    shapiro_test = stats.shapiro(residuals)
    print(f'Shapiro-Wilk statistics: {shapiro_test[1]}')
    if shapiro_test[1] < 0.05:
        print("Residuals are not normally distributed (Reject H0)")
    else:
        print("Residuals are normally distributed (Fail to reject H0)")
    
    # 3. Check for Autocorrelation
    dw_stat = durbin_watson(residuals)
    print(f'Durbin-Watson statistic: {dw_stat}')
    
    # 4. Multicollinearity (VIF)
    X_train_preprocessed = model.named_steps['preprocessor'].transform(X_test)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_train_preprocessed_df.columns
    vif_data['VIF'] = [variance_inflation_factor(X_train_preprocessed_df.values, i) 
                       for i in range(X_train_preprocessed_df.shape[1])]
    print(vif_data)
    
    # 5. Homoscedasticity
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=fitted, y=np.sqrt(np.abs(residuals)))
    plt.xlabel('Fitted values')
    plt.ylabel('Sqrt(|Residuals|)')
    plt.title('Scale-Location Plot')
    plt.show()


    X_train_preprocessed_df_const = sm.add_constant(X_train_preprocessed_df)
    bp_test = het_breuschpagan(residuals, X_train_preprocessed_df_const)
    lm_statistic, lm_pvalue, f_value, f_pvalue = het_breuschpagan(residuals, X_train_preprocessed_df_const)

    print(f'Breusch-Pagan test statistic: {lm_pvalue}')
    if lm_pvalue < 0.05:
        print('The null hypothesis of homoscedasticity is rejected. There is evidence of heteroscedasticity.')
    else:
        print('The null hypothesis of homoscedasticity cannot be rejected. No evidence of heteroscedasticity.')

# Inverse Transformation on Target Variable Function
def inverse_transform(y, transform=None, scaler=None):
    if transform == 'log':
        return np.exp(y)
    elif transform == 'standardize':
        return scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    else:
        return y
  
# Build Regression Model Function
def regression(df, features, target, transform=None, model_type='Linear', degree = None, df_name=''):
    print(f"\n--- {model_type} Regression for {df_name} ---")

    # Clean and standardize column names to ensure consistency
    df.columns = df.columns.str.strip().astype(str)
    features = [feature.strip() for feature in features]
    target = target.strip()
    
    X = df[features]
    y = df[target]
    
    scaler = None

    # Apply target transformation 
    if transform == 'log':
        y = np.log(y)
    elif transform == 'standardize':
        scaler = StandardScaler()
        y = scaler.fit_transform(y.values.reshape(-1,1)).flatten()
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
    # Preprocessing
    num_features = X.select_dtypes(include=['float64', 'int64']).columns

    
    Preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features)
        ],
        remainder='passthrough'
    )
    
    if model_type == 'Linear' or model_type == 'Polynomial':
        model = LinearRegression()
    elif model_type == 'Ridge':
        model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    elif model_type == 'Lasso':
        model = LassoCV(cv=5)
    
       
    if model_type == 'Polynomial':
        pipeline = Pipeline([
            ('preprocessor', Preprocessor),
            ('poly', PolynomialFeatures(degree=degree)),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
        ('preprocessor', Preprocessor),
        ('model', model)])
    
    # Fit Model
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)

    # Inverse Transformation
    y_pred_inv = inverse_transform(y_pred,transform, scaler)
    y_test_inv = inverse_transform(y_test,transform, scaler)
    
    
    # Evaluate
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    print(f"R² Score: {r2}")
    

    # Training Set
    y_train_pred = pipeline.predict(X_train)
    y_train_pred_inv = inverse_transform(y_train_pred,transform, scaler=scaler)
    y_train_inv = inverse_transform(y_train,transform, scaler=scaler)
    train_rmse = np.sqrt(mean_squared_error(y_train_inv, y_train_pred_inv))
    print(f'Training RMSE: {train_rmse:.4f}')
    
    return {
        'DataFrame': df_name,
        'Transformation': transform,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'training_RMSE': train_rmse
    }


def reg_assumption_check(df, features, target, transform=None, model_type='Linear', degree = None, df_name=''):
    print(f"\n--- {model_type} Regression for {df_name} ---")
    
    X = df[features]
    y = df[target]
    
    
    scaler = None

    # Apply target transformation 
    if transform == 'log':
        y = np.log(y)
    elif transform == 'standardize':
        scaler = StandardScaler()
        y = scaler.fit_transform(y.values.reshape(-1,1)).flatten()
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
    # Preprocessing
    num_features = X.select_dtypes(include=['float64', 'int64']).columns

    
    Preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features)
        ],
        remainder='passthrough'
    )
    
    if model_type == 'Linear' or model_type == 'Polynomial':
        model = LinearRegression()
    elif model_type == 'Ridge':
        model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    elif model_type == 'Lasso':
        model = LassoCV(cv=5)
     
    if model_type == 'Polynomial':
        pipeline = Pipeline([
            ('preprocessor', Preprocessor),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
        ('preprocessor', Preprocessor),
        ('model', model)])
    
    # Fit Model
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)

   
    # Check Assumptions
    check_assumptions(pipeline, X_test, y_test, y_pred)

#%% Modelling 

##### Define Features and Target ###


#%%

cp_dummies = pd.get_dummies(cp_data_cleaned,columns=['ward'],drop_first=True)
cp_dummies.columns = cp_dummies.columns.astype(str)

controls = ['bathrm','rooms','bedrm',
            'fireplaces','year','ward_2', 
            'ward_3','ward_4', 'ward_5', 
            'ward_6', 'ward_7', 'ward_8']


controls = ['bathrm', 'rooms', 'bedrm', 'fireplaces', 'ward', 'year', 'median_gross_income']

total = ['total_crime_count']
methods = ['method_gun', 'method_knife', 'method_others']
shifts = ['shift_day', 'shift_midnight', 'shift_evening']
offenses = ['violent_crime_count', 'property_crime_count']

### Offenses #####

feature_sets = [controls, controls+total, controls+offenses, controls+methods, controls+shifts]


target = 'price'


transformations = [None] 
model_types = ['Linear', 'Polynomial', 'Ridge', 'Lasso']

degrees = [2]

results = []

for feature_set in feature_sets:
    for transform in transformations:
        for model_type in model_types:
            if model_type == 'Polynomial':
                for degree in degrees:
                    result = regression(cp_dummies, feature_set, target, transform, model_type, degree, df_name=f"Final Dataframe with {transform or 'No'} Transform on y")
                    results.append(result)
            else:
                result = regression(cp_dummies, feature_set, target, transform, model_type, None, df_name=f"Final Dataframe with {transform or 'No'} Transform on y")
                results.append(result)

reg_assumption_check(cp_dummies, feature_set, target, model_type='Polynomial', degree=2)

#%% Compile Results
results_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(results_df)

#%%

### Ward as categorical Variable ###
### Offense Features ####



features = controls+offenses
features


target = 'price'


transformations = [None] 
model_types = ['Linear', 'Polynomial']

degrees = [2, 3]

results = []


for transform in transformations:
    for model_type in model_types:
        if model_type == 'Polynomial':
            for degree in degrees:
                result = regression(cp_dummies, features, target, transform, model_type, degree, df_name=f"{df_name} with {transform or 'No'} Transform on Target Variable")
                results.append(result)
        else:
            result = regression(cp_dummies, features, target, transform, model_type, None, df_name=f"{df_name} with {transform or 'No'} Transform on Target Variable")
            results.append(result)

#%% Compile Results
results_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(results_df)
#%%

reg_assumption_check(cp_dummies, features, target, transform= 'log', degree=2)


#%%

features = cp_dummies.drop(["price"], axis=1).columns

dataframes = {
    'With dummy variables': cp_dummies
}

transformations = [None, 'standardize', 'log'] 

model_types = ['Ridge', 'Lasso']

results = []

for df_name, df in dataframes.items():
    for transform in transformations:
        for model_type in model_types:
            result = regression(df, features, target, transform, model_type, df_name=f"{df_name} with {transform or 'No'} Transform on Target Variable")
            results.append(result)


#%%
results_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(results_df)
