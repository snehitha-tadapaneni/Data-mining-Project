#%%[markdown]
## Project Overview

### Topic
# **Analysis of Crime Rates on Residential Property in Washington DC in 2018**

### Team Members
# - Palrecha Sayam Mukesh
#- Snehitha Tadapaneni
#- Amrutha Jayachandradhara
#- Annie Cheng

### Course
#**Introduction to Data Mining: DS 6103**  
#**Instructor**: Prof. Ning Rui  
#**TA**: Parameshwar Bhat

### SMART Questions
#1. To what extent do neighborhood crime rates correlate with residential property values across Washington DC, using the combined analysis of the Housing Price dataset and open data crime dataset?
#   
#2. How accurately can violent crime rates classify neighbourhoods in DC, into different 3 different housing price tiers as low, medium and high?
# 

### Datasets
#- **DC Crime Dataset**: [DC Open Data Crime Dataset](https://opendata.dc.gov/datasets/c5a9f33ffca546babbd91de1969e742d_6/explore?location=38.903935%2C-77.012050%2C10.79)
#- **House Pricing Dataset**: [Kaggle DC Residential Properties](https://www.kaggle.com/datasets/christophercorrea/dc-residential-properties?select=raw_address_points.csv)

#%%
# Importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import f_oneway
import geopandas as gpd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


#%%[markdown]
## Data Preparation
# We have merged and aggregated both datasets based on `census_tract` and offense counts. 
# The final dataset contains 31 columns, including house characteristics (e.g., price, rooms, bathrooms) and detailed crime-related features.
# Specifically, the crime-related deatures include each offense category (e.g., ARSON, BURGLARY, HOMICIDE, THEFT), 
# method of committing crimes (e.g., GUN, KNIFE), and shift (e.g., DAY, NIGHT).
# This comprehensive dataset enables a thorough analysis of the relationship between housing attributes and crime rates.


#%%
#############################
# Step 1: Load and preprocess crime data
#############################

# Load crime data
dc_crime = pd.read_csv('dc_crime.csv', index_col = 0) 

# Remove rows where 'start_year' is missing
dc_crime = dc_crime.dropna(subset=['start_year'])

# Convert 'start_year' to integer for consistency
dc_crime['start_year'] = dc_crime['start_year'].astype(int)

# Keep rows where 'census_tract' is not null and convert it to integer
dc_crime = dc_crime[dc_crime['census_tract'].notnull()]
dc_crime['census_tract'] = dc_crime['census_tract'].astype(float).astype(int)


#%%
#############################
# Step 2: Create dummy variables for categorical columns
#############################

# Generate dummy variables for offense types, methods, and shift
offense_dummies = pd.get_dummies(dc_crime[['start_year', 'census_tract', 'offense']], columns=['offense'])
method_dummies = pd.get_dummies(dc_crime[['start_year', 'census_tract', 'method']], columns=['method'])
shift_dummies = pd.get_dummies(dc_crime[['start_year', 'census_tract', 'shift']], columns=['shift'])

#%%
#############################
# Step 3: Group data by year and census tract
#############################

# Sum dummy variables grouped by 'start_year' and 'census_tract'
offense_grouped = offense_dummies.groupby(['start_year', 'census_tract']).sum().reset_index()
method_grouped = method_dummies.groupby(['start_year', 'census_tract']).sum().reset_index()
shift_grouped = shift_dummies.groupby(['start_year', 'census_tract']).sum().reset_index()

# Filter data for years 2014 and later
offense_grouped = offense_grouped[offense_grouped['start_year'] >= 2014]
method_grouped = method_grouped[method_grouped['start_year'] >= 2014]
shift_grouped = shift_grouped[shift_grouped['start_year'] >= 2014]

#%%
#############################
# Step 4: Merge grouped data into a combined dataset
#############################

# Merge offense, method, and shift data into a single dataframe
crime_census_combined = offense_grouped.merge(
    method_grouped, 
    on=['start_year', 'census_tract']
).merge(
    shift_grouped, 
    on=['start_year', 'census_tract']
)

# Convert 'census_tract' to object type for consistency with housing data
crime_census_combined['census_tract'] = crime_census_combined['census_tract'].astype(object)
crime_census_combined['start_year'] = crime_census_combined['start_year'].astype(int)

#%%
#############################
# Step 5: Load and preprocess housing data
#############################

# Load housing data
dc_housing = pd.read_csv("tract_house_101.csv", index_col = 0)

# Drop unnecessary columns from the housing dataset
columns_to_drop = [
    'HF_BATHRM','HEAT','AC','AYB','YR_RMDL','EYB','STORIES','QUALIFIED','SALE_NUM','GBA','BLDG_NUM','STYLE','STRUCT','GRADE','CNDTN',
 'EXTWALL','ROOF','INTWALL', 'USECODE','LANDAREA','GIS_LAST_MOD_DTTM', 'SOURCE','CMPLX_NUM','LIVING_GBA','FULLADDRESS','CITY','STATE',
 'ZIPCODE','NATIONALGRID','LATITUDE','LONGITUDE','ASSESSMENT_NBHD','ASSESSMENT_SUBNBHD','CENSUS_BLOCK','SQUARE','X', 'Y','QUADRANT', 'TRACT',
 'GEOID', 'P0010001','P0010002','P0010003','P0010004','P0010005','P0010006','P0010007','P0010008','OP000001','OP000002','OP000003',
 'OP000004','P0020002','P0020005','P0020006','P0020007','P0020008','P0020009','P0020010','OP00005','OP00006','OP00007','OP00008',
 'P0030001','P0030003','P0030004','P0030005','P0030006','P0030007','P0030008','OP00009','OP00010','OP00011','OP00012','P0040002','P0040005',
 'P0040006','P0040007','P0040008','P0040009','P0040010','OP000013','OP000014','OP000015','OP000016','H0010001','H0010002','H0010003',
 'SQ_MILES','Shape_Length','Shape_Area','FAGI_TOTAL_2010','FAGI_MEDIAN_2010','FAGI_TOTAL_2013','FAGI_MEDIAN_2013','FAGI_TOTAL_2011','FAGI_MEDIAN_2011','FAGI_TOTAL_2012',
 'FAGI_MEDIAN_2012','FAGI_TOTAL_2015','FAGI_MEDIAN_2015'
]
dc_housing = dc_housing.drop(columns=columns_to_drop)

# Standardize column names to lowercase for consistency
dc_housing.columns = dc_housing.columns.str.lower()

# Convert 'census_tract' to object type for consistency
dc_housing['census_tract'] = dc_housing['census_tract'].astype(object)

#%%
#############################
# Step 6: Merge crime and housing data
#############################

# Merge housing and crime data based on 'census_tract' and year
cp_data = dc_housing.merge(
    crime_census_combined, 
    left_on=['saleyear', 'census_tract'], 
    right_on=['start_year', 'census_tract']
)

#%%
#############################
# Step 7: Save the final merged dataset
#############################

# Export the combined data to a CSV file
#cp_data.to_csv('final_return_new.csv', index=False)

#%%[markdown]
# Now, we have our final dataset saved as final_return_new.csv. Let's proceed with our Exploration!
## Data Exploring and Cleaning
#%%
# Reading the dataset into cp_data
cp_data = pd.read_csv("final_return_new.csv")
# %%
# Look at the first 5 rows of the dataset
cp_data.head()

# Look at the last 5 rows of the dataset
cp_data.tail()

# Shape of our merged dataset
cp_data.shape

# Checking columns in our merges dataset
cp_data.columns

# Checking the datatypes
cp_data.info()

# Statistics of the data
cp_data.describe()

#%%
# Checking for null values/ missing values
cp_data.isnull().sum()
# A heatmap to visualise the missing data points if any
sns.heatmap(cp_data.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values in Dataset")
plt.show()

#%%[markdown]
#### We can see missing values in num_units, price and kitchens. Let's handle them!
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
# Our final cleaned data has columns
print(cp_data.columns)

#%%[markdown]
# Now, that we have cleaned our dataset. Let's Explore and learn more about our features.

#

## Data Visualization: Univariate Analysis
# <br>
# Distribution for all numerical features and the target variable
# %%
# Histograms for numerical features
num_cols = ['bathrm', 'rooms', 'fireplaces', 'bedrm', 'year', 'ward', 'median_gross_income', 'offense_arson', 'offense_assault w/dangerous weapon', 'offense_burglary', 'offense_homicide', 'offense_motor vehicle theft', 'offense_robbery', 'offense_sex abuse', 'offense_theft f/auto', 'offense_theft/other', 'method_gun', 'method_knife', 'method_others', 'shift_day', 'shift_evening', 'shift_midnight']
cp_data[num_cols].hist(figsize=(10, 12), layout=(6, 4), edgecolor='black')
plt.suptitle('Distributions of Numerical Features')
plt.show()

# Analysing the target variable - plotting a distribution to understand price
def plot_target_variable(df,cols_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['cols_name'], kde=True)
    plt.title("Distribution of Housing Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.show()

# %%[markdown]
# As the price distribution is highly skewed, lets look at the outliers in the target variable.
# Boxplot for detecting outliers in price
plt.figure(figsize=(10, 6))
sns.boxplot(x=cp_data['price'])
plt.title("Boxplot of Housing Prices")
plt.show()

# As we can see, there are price values too high that can impact the models ability to understand and interpret the the data. To make our analysis, more specific and less comples, we are sticking to certain pricing values.
# We will only interpret residential property values untill 1500000$. Let's consider all the other points as outliers and remove them.

#%%
# Removing the outliers from the target variable: price

cp_data_cleaned = cp_data.copy()

q1, q3 = np.percentile(cp_data['price'], 25), np.percentile(cp_data['price'], 75)

iqr = q3-q1
lower = q1 - 1.5*iqr
upper = q3 + 1.5*iqr

# Removing the outliers
cp_data_cleaned = cp_data_cleaned[(cp_data_cleaned['price'] >= lower) & (cp_data_cleaned['price'] <= upper)]

print("New Shape: ", cp_data_cleaned.shape)
cp_data_cleaned.info()
#%%
# Frequency of each method type: Plot for price vs method types
methods = ['method_gun', 'method_knife', 'method_others']
method_sums = cp_data_cleaned[methods].sum()

plt.figure(figsize=(8, 6))
method_sums.plot(kind='bar', color=['skyblue', 'orange', 'green'], alpha=0.8)
plt.title('Frequency of Each Method Type', fontsize=14)
plt.xlabel('Method Type', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=0)
plt.show()

# Distribution of price after removing the outliers
plt.figure(figsize=(8, 6))
sns.histplot(cp_data_cleaned['price'], kde=True, color='purple', bins=30)
plt.title('Distribution of Price', fontsize=14)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()
#%%[markdown]
# The price distribution(after removing the outliers) appears to follow a slightly right-skewed distribution (positive skewness).
# This tells us that the majority of prices are concentrated towards the lower and middle ranges, while fewer higher prices create a longer tail on the right. Normalizing or scaling the data would be required!

#
#%%[markdown]
## Data Visualization: Bivariate Analysis
# <br>
# Correlation heatmap for all Numerical Variables
# %%
# Heatmap to understand relationship between price and other numerical variables to plot a heatmap and understand the correlation of the features with the target variables

# Plot the heatmap
def plot_heatmap(df):
    # Selected only numerical features from the dataset
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_df = df[numerical_cols]
    # Compute the correlation matrix for the numerical features 
    corr = numerical_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('Correlation Heatmap')
    plt.show()
#%%[markdown]
# *Price Correlations Obervations*:
# <br> 1. The variable PRICE has a moderate positive correlation with CENSUS_TRACT (correlation ~0.54), showing some geographical influence on prices. Also, price has a positive correlation with median gross income of the households.
# <br> 2. BATHRM, ROOMS, and FIREPLACES show mild positive correlations with PRICE, indicating that these features might drive higher property values.
# <br> 3.  METHOD_GUN and other offense-related variables have a negative correlation with PRICE, implying that crime rates might negatively impact property values.
# <br><br>
# Let us dig deep to support our observations.

# Plot for Price vs Property Attributes (bathrooms, rooms, bedrooms, fireprices)
#%%
def make_boxplot(df,var_x,var_y,name_x):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[var_x], y=df[var_y], data=df)
    plt.title("Housing Prices Based on " + name_x)
    plt.xlabel(name_x)
    plt.ylabel("Price")
    plt.show()


# Prices vs rooms
make_boxplot(cp_data_cleaned,'rooms','price', 'Number of Rooms')

# Prices vs bathrooms
make_boxplot(cp_data_cleaned,'bathrm','price', 'Number of Bathrooms')

# Prices vs bed room
make_boxplot(cp_data_cleaned,'bedrm','price', 'Number of Bedrooms')

# Prices vs Fireplaces
make_boxplot(cp_data_cleaned,'fireplaces','price', 'Number of Fireplaces')


# %% [markdown]
# Now let's visualize Geographical Variation of Property Price (Ward)
# Aggregate the median house price for each census tract
# Prices vs ward

make_boxplot(cp_data_cleaned,'ward','price', 'Ward')

# %% [markdown]
# Now let's visualize Geographical Variation of Property Price (Ward)
# Aggregate the median house price for each census tract
ward_house = cp_data_cleaned.groupby('ward').agg(
    price_median=('price', 'median'), 
    price_mean=('price', 'mean')).reset_index()

# Create a bar plot: Ward vs Price
def barplot_ward_price(df):
    grouped_data = df.groupby('ward').agg({
    'price': 'median'}).reset_index()

    # Set the plotting style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grouped_data, x=df['ward'], y=df['price'], palette='coolwarm')
    plt.title('Ward vs Median Price')
    plt.xlabel('Ward')
    plt.ylabel('Median Price')
    plt.tight_layout()
    plt.show()

barplot_ward_price(cp_data_cleaned)

# Load ward shapefile
ward_map = gpd.read_file('Wards_from_2022.shp')

def make_map(df, df_map, area, var):

    # Merge shapefile data with housing data for wards
    df_merged = df_map.merge(df, left_on=area.upper(), right_on=area, how='left')
    # Data Visualization of Median House Price by Ward on Map
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Specify map data visualization you want 
    if var == 'price_median':
        df_merged.plot(column=var, cmap='Blues', linewidth=1.0, ax=ax, edgecolor='0.5', legend=True)
    elif var == 'total_crime':
        df_merged.plot(column=var, cmap='OrRd', linewidth=1.0, ax=ax, edgecolor='0.5', legend=True)
    
    # Add labels to Ward    
    if area == 'ward':
        for idx, row in df_merged.iterrows():
            centroid = row['geometry'].centroid
            price_median = row[var]

            # Set color based on the median price
            text_color = 'white' if price_median > 400000 else 'black'
            # Add the ward label
            plt.text(centroid.x, centroid.y, str(row['WARD']), fontsize=15, ha='center', color=text_color)
    
    if var == 'price_median':
        plt.title(f'Median Property Value by {area}')
    elif var == 'total_crime':
        plt.title(f'Total Crime Count by {area}')
    plt.show()

make_map(ward_house, ward_map, 'ward', 'price_median')
# plt.savefig('house_ward_map.png')

#%% [markdown]

# Data visualization on time variation of housing price by year

make_boxplot(cp_data_cleaned,'year','price', 'Year')

yr_median = cp_data_cleaned.groupby('year')['price'].median().reset_index()


# Line plot
def make_lineplot(df, col_name, target):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=col_name, y=target, data=df)
    plt.title("Yearly Trend of Median Housing Price")
    plt.show()

make_lineplot(yr_median, 'year', 'price')

#%% [markdown]
# The boxplot shows that while the overall distribution of housing prices appears similar across different years. 
# Despite the similarity in distribution, the median housing prices display a noticeable trend over the years. 

#%% [markdown]
### Crime Related Features vs Housing Price
#### Housing Price vs Total Crime Counts
# Now we will visualize how tract level total crime counts can affect property price.
#
# Calculate the total crime counts
cp_data_cleaned['total_crime_count'] = cp_data_cleaned[
    ['offense_assault w/dangerous weapon', 'offense_homicide', 
     'offense_robbery', 'offense_sex abuse', 'offense_arson',
    'offense_burglary', 'offense_motor vehicle theft', 'offense_theft f/auto', 
    'offense_theft/other']].sum(axis=1)

# Scatter plot for total crimes vs price
def make_scatter(df, var_x, var_y, name_x):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[var_x], df[var_y], alpha=0.6)
    plt.title(f'Scatter Plot: {name_x} vs Price')
    plt.xlabel(name_x)
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
   
make_scatter(cp_data_cleaned, 'total_crime_count', 'price', 'Total Crime Counts')

#%% [markdown]
# In scatter plot, we cannot observe a clear patterns between total crime counts and price.
# To further investigate, we will visualize the distribution of crimes and property values by census tract on map. 
# This will allow us to explore whether neighborhoods with higher crime occurrences tend to have lower property values.

#%%

# Aggregate the total crime counts for each neighborhood
tract_crime = cp_data_cleaned[
    ['census_tract', 
     'year', 
     'total_crime_count']
     ].drop_duplicates(subset=['census_tract', 'year']).groupby('census_tract').agg(
            total_crime = ('total_crime_count', 'sum'))

# Aggregate the median house price for each census tract
tract_house = cp_data_cleaned.groupby('census_tract').agg(
    price_median=('price', 'median')).reset_index()
tract_house

tract_map = gpd.read_file('Census_Tracts_in_2010.shp')

# Clean up tract variable and convert into integer
tract_map['TRACT'] = tract_map['TRACT'].str.lstrip('0').astype(int)
tract_map.rename(columns = {'TRACT': 'CENSUS_TRACT'}, inplace = True)

make_map(tract_house, tract_map, 'census_tract', 'price_median')
make_map(tract_crime, tract_map, 'census_tract', 'total_crime')

#%% [markdown]

# By visualizing from the maps, we can observe that areas with higher crime counts generally exhibit lower property values, 
# suggesting a possible negative impact of crime concentration on property values.

#%%[markdown]
# # Scatter plot: All method types(gun, knife, others) vs housing prices
#%%
def make_scatter_method(df, methods):

    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'yellow']

    for i in range(len(methods)):


        # Plot for each method
        plt.scatter(
            df.loc[df[methods[i]] == 1, 'price'],
            df.loc[df[methods[i]] == 1].index,
            color=colors[i], label='Gun', alpha=0.6
        )


    # Scatter plot: method types vs price
    plt.title('Scatter Plot: Price vs Method Types', fontsize=14)
    plt.xlabel('Price', fontsize=12)
    plt.ylabel('Index (Observations)', fontsize=12)
    plt.legend(title='Method Type')
    plt.grid(alpha=0.5, linestyle='--')
    plt.show()

make_scatter_method(cp_data_cleaned, ['method_gun', 'method_knife', 'method_others'])


#%%[markdown]
# From the above scatter plot, we can see that there are no Method_Others influencing the price values. 
# Let us perform a statistical test(Spearman Correlation) to check the relationship between the price and the method types and prove our point.
#%%

def spearman_corr(df, target, feature):

    # Calculate Spearman correlation between 'price' and each 'method' type
    corr, p_value = spearmanr(df[target], df[feature])
    # Display the results
    print(f"Spearman Correlation for {feature}: {corr}, p-value: {p_value}")

spearman_corr(cp_data_cleaned, 'price', 'method_gun')
spearman_corr(cp_data_cleaned, 'price', 'method_knife')
spearman_corr(cp_data_cleaned, 'price', 'method_others')

#%%[markdown]
# Let us state our hypothesis,<br>
# Null Hypothesis (H₀): There is no monotonic relationship between price and method_GUN, method_KNIFE, method_OTHERS.
# Alternative Hypothesis (H₁): There is a significant monotonic relationship between price and method_GUN, method_KNIFE, method_OTHERS.

# 1. For method_GUN:<br>
# Interpretation: Since the p-value is 0.0 (which is less than 0.05), we reject the null hypothesis, indicating a significant monotonic relationship between price and method_GUN. This suggests that as the price increases or decreases, there is a tendency for the frequency of gun-related incidents to change in a monotonic manner.

# 2. For method_KNIFE:<br>
# Interpretation: The p-value is also 0.0, which is less than 0.05, so we reject the null hypothesis, indicating a significant monotonic relationship between price and method_KNIFE. This suggests that there is a weak but significant trend of knife-related incidents associated with price changes.

# 3. For method_OTHERS:<br>
# Interpretation: Since the p-value is 0.976e, we reject the null hypothesis, indicating a significant monotonic relationship between price and method_OTHERS. This suggests that changes in price does significantly affect the occurrence of incidents categorized as "Others."
#<br>


# Scatter plot between crime categories vs the price distribution
# %%
# Aggregate crime counts as violent crime and property crime
cp_data_cleaned['violent_crime_count'] = cp_data_cleaned[['offense_assault w/dangerous weapon', 'offense_homicide', 'offense_robbery', 'offense_sex abuse']].sum(axis=1)

cp_data_cleaned['property_crime_count'] = cp_data_cleaned[['offense_arson', 'offense_burglary', 'offense_motor vehicle theft', 'offense_theft f/auto', 'offense_theft/other']].sum(axis=1)

# Scatter plot for crimes vs price
def plot_crime_price(df,):
    df['violent_crime_count'] = df[['offense_assault w/dangerous weapon', 'offense_homicide', 'offense_robbery', 'offense_sex abuse']].sum(axis=1)

    df['property_crime_count'] = df[['offense_arson', 'offense_burglary', 'offense_motor vehicle theft', 'offense_theft f/auto', 'offense_theft/other']].sum(axis=1)

    plt.figure(figsize=(10, 6))
    plt.scatter(df['violent_crime_count'], df['price'], color='red', alpha=0.6, label='Violent Crimes')
    plt.scatter(df['property_crime_count'], df['price'], color='blue', alpha=0.6, label='Property Crimes')
    plt.title('Scatter Plot: Violent and Property Crimes vs Price')
    plt.xlabel('Crime Count')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

plot_crime_price(cp_data_cleaned)
#%%[markdown]
# As, we can see the above scatter plot is too complex to understand.
# Let us aggregate the data based on census tract and plot the violent and property crime values for more clarity.

#%%
# Aggregate data by census tract
tract_data = cp_data_cleaned.groupby('census_tract').agg({
    'violent_crime_count': 'sum',
    'property_crime_count': 'sum',
    'price': 'mean'  # Average price per tract
}).reset_index()

# Scatter plot for aggregated crime counts vs average price
def aggcrime_price(df):
    tract_data = df.groupby('census_tract').agg({
    'violent_crime_count': 'sum',
    'property_crime_count': 'sum',
    'price': 'mean'}).reset_index()
    plt.figure(figsize=(10, 6))
    plt.scatter(tract_data['violent_crime_count'], tract_data['price'], color='red', alpha=0.6, label='Violent Crimes')
    plt.scatter(tract_data['property_crime_count'], tract_data['price'], color='blue', alpha=0.6, label='Property Crimes')
    plt.title('Crime Counts vs Average Price by Census Tract')
    plt.xlabel('Aggregated Crime Count')
    plt.ylabel('Average Price ($)')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
aggcrime_price(cp_data_cleaned)
#%%
# Compute correlation coefficients: Price vs violent crime vs property crime
correlation_matrix = cp_data_cleaned[['price', 'violent_crime_count', 'property_crime_count']].corr()

# Display correlation matrix
def corr_plot(df):
    correlation_matrix = df[['price', 'violent_crime_count', 'property_crime_count']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix: Price vs Crime Counts')
    plt.show()

corr_plot(cp_data_cleaned)
#%%[markdown]
# From the above heatmap between Violent Crime, Prperty Crime and Price values, we can say:<br>
# 1. Violent crime has a stronger and negative impact on property prices compared to property crimes.<br>
# 2. Property crimes are weakly related to prices, indicating they may not be a strong factor influencing property values in the dataset.<br>
# 3. The moderate positive correlation between violent and property crimes suggests that crime types are somewhat related in occurrence.<br>

#%%[Markdown]
## Modelling Techniques

#### Smart Question 1: For regression

##### Light GBM

# The first model is trained using only `total_crime_count` and controlling varuiables,
# while the second one is trained with all detailed crime features (`violent_crime_count`,
# `property_crime_count`,`method_gun`, `method_knife``, 'method_others', 'shift_day',
# 'shift_midnight','shift_evening')

#%%

### Model 1: Total Crime Counts ###

# Define features
controls = ['bathrm', 'rooms', 'bedrm', 'fireplaces', 'year', 'ward', 'median_gross_income']
total = ['total_crime_count']
features = controls + total

X = cp_data_cleaned[features]
y = cp_data_cleaned['price']

# Clean up column names
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
cat_features = ['ward']
X[cat_features] = X[cat_features].astype('category')

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  

# Define initial model parameters
initial_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': 42
}

# Initialize LGBMRegressor
model = LGBMRegressor(**initial_params)

# Fit the model
model.fit(
    X_train, y_train,
    eval_metric='rmse',
    categorical_feature=cat_features
)

# Test RMSE and R2
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)
print(f'Initial Test RMSE: {test_rmse:.4f}')
print(f'Initial Test R²: {test_r2:.4f}')

#%% 
######### Hyperparameter Tuning ################# 

# Define parameter search space
param_grid = {
    'learning_rate': [0.03, 0.05, 0.07],
    'num_leaves': [20, 27, 31],
    'max_depth': [7, 10, 15],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.3]
}

# Initialize LGBMRegressor
lgbm = LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    verbosity=-1,
    random_state=42
)

# Initialize a KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=kf,
    verbose=1,
    n_jobs=1
)

# Perform Grid Search
grid_search.fit(X_train, y_train,categorical_feature=cat_features)

# Extract the best parameters and best CV RMSE
best_params = grid_search.best_params_
best_rmse = -grid_search.best_score_
print('Best Parameters:', best_params)
print(f'Best CV RMSE: {best_rmse:.4f}')

#%%

# best_params = {'learning_rate': 0.07, 'max_depth': 15, 'num_leaves': 31, 'reg_alpha': 0, 'reg_lambda': 0.1}

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

# Initialize the final model with the best parameters
final_model = LGBMRegressor(**model_params)

# Fit the final model on training data
final_model.fit(
    X_train, y_train,
    eval_metric='rmse',
    categorical_feature=cat_features
)


####### Final Model Evaluation ############

# Calculate Test RMSE and R2
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

# %% [markdown]

### Model 2: All Featrues ###

# Define features
features = ['violent_crime_count', 'property_crime_count',
            'method_gun', 'method_knife', 'method_others',
            'shift_day', 'shift_midnight', 'shift_evening']


X = cp_data_cleaned[features]
y = cp_data_cleaned['price']

# Clean up column name
X.columns = X.columns.str.replace(' ', '_')

# Define categorical features and convert their data types to 'category'
cat_features = ['ward']
X[cat_features] = X[cat_features].astype('category')

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  

# Initialize inital parameters
initial_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'random_state': 42
}

initial_model = LGBMRegressor(**initial_params)

# Fit initial model on training data
initial_model.fit(
    X_train, y_train,
    categorical_feature=cat_features,
    eval_metric='rmse'
)

# Calculate Test Data RMSE and R2
y_test_pred_initial = initial_model.predict(X_test, num_iteration=initial_model.best_iteration_)
test_rmse_initial = mean_squared_error(y_test, y_test_pred_initial, squared=False)
test_r2_initial = r2_score(y_test, y_test_pred_initial)

# Print out results
print(f"Initial Model Test RMSE: {test_rmse_initial:.4f}")
print(f"Initial Model Test R²: {test_r2_initial:.4f}")

#%% 
########## Hyperparameter Tuning ################

# Define parameter search space
param_grid = {
    'learning_rate': [0.01, 0.05, 0.07, 0.1],
    'num_leaves': [20, 31, 62],
    'max_depth': [7, 10, 15],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Initialize LGBMRegressor
lgbm = LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    categorical_feature=X_train.columns.get_loc('ward'),
    verbosity=-1,
    random_state=42

)

# Initialize kFold cross validation
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

# Perform grid search to find the best parameters
grid_search.fit(X_train, y_train, categorical_feature=cat_features)

# Extract the best parameters and best CV RMSE
best_params = grid_search.best_params_
best_rmse = -grid_search.best_score_
print('Best Parameters: ', best_params)
print(f'Best CV RMSE: {best_rmse:.4f}')

#%%
#best_params =  {'learning_rate': 0.1, 'max_depth': 15, 'num_leaves': 62, 'reg_alpha': 0.5, 'reg_lambda': 0.1}

model_params ={
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    **best_params}


# Initialize the final model with the best parameters
final_model = LGBMRegressor(**model_params)


# Fit the final model
final_model.fit(
    X_train, y_train,
    eval_metric='rmse',
    categorical_feature = X_train.columns.get_loc('ward')
)

# Initialize cross validation
lgbm_cv = LGBMRegressor(**model_params)

# Perform cross-validation 
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

######### Final Model Evaluation #############

# Calculate Test Data RMSE and R2
y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration_)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)

# Calculate Cross Validation RMSE and R2 
cv_rmse_best = -cv_scores_best['test_neg_root_mean_squared_error']
cv_r2_best = cv_scores_best['test_r2']

# Print out results
print(f'Final Model Test RMSE: {test_rmse:.4f}')
print(f'Final Model Test R²: {test_r2:.4f}\n')

print(f'Average Cross-Validation RMSE: {cv_rmse_best.mean():.4f}')
print(f'Average Cross-Validation R²: {cv_r2_best.mean():.4f}')

# Actual vs. Predicted Values Plot 
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (All Features)')
plt.show()

#%% 
# Final Feature Importance Analysis
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

#%% [markdown]
# __Model Evaluation__: <br>
# Comparing R2 and RMSE, the model with detailed crime features (R2 = 76.22%, RMSE = 137162.89) shows a marginal improvement over the model with only total crime counts (R2 = 75.62%, RMSE = 138894.19).
# The R2 score indicates that the detailed crime features model explains 76.22% of the variation in property prices, while the total crime count model explains 75.62%.
# However, both models perform similarly, as seen in the actual vs. predicted plots, where predictions align closely with the ideal line. 
# Both models exhibit slightly greater variance at the higher end of property prices, with the detailed model showing slightly better alignment overall.
# 
# __Feature Importance__: <br>
# Across both models, features like median gross income and ward consistently rank high in both split and gain importance. 
# This highlights their strong predictive power and confirms that these factors are key determinants of residential property values.
# Property-specific attributes such as bathrooms, bedrooms, and fireplaces further reinforce the dominance of these variables over crime-related data in predicting housing prices.
#
# __Insights__:  <br>
# From the model performance and feature importance, property attributes and socioeconomic factors are the strongest determinants of residential property values in Washington, DC.
# While crime does influence property prices, its impact is modest compared to these primary factors.

#%%

##### Random Forest Regressor



#%%[Markdown]
#### Smart Question 2: For Classification

##### Random Forest Classifier

#%%
from sklearn.model_selection import train_test_split

cp_data_cleaned['price_category'] = pd.qcut(cp_data['price'], q=3, labels=[0, 1, 2])

X = cp_data_cleaned.drop(columns=['price_category', 'price', 'offense_arson', 'offense_assault w/dangerous weapon',
       'offense_burglary', 'offense_homicide', 'offense_motor vehicle theft',
       'offense_robbery', 'offense_sex abuse', 'offense_theft f/auto',
       'offense_theft/other'])

y = cp_data_cleaned['price_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

rf = RandomForestClassifier(random_state=42, max_depth=20, max_features='sqrt', min_samples_leaf=2, n_estimators=100, min_samples_split=15)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluate
accuracy = rf.score(X_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%%
# Feature importance

# Get feature importance
feature_importance = rf.feature_importances_

# Create a DataFrame for better visualization
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest')
plt.gca().invert_yaxis()  # Reverse the order for readability
plt.show()

#%%
# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=5,  # 5-fold cross-validation
                           scoring='accuracy', 
                           n_jobs=-1, 
                           verbose=2)

grid_search.fit(X_train, y_train)

# Get the best parameters and accuracy
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score:.2f}")

#%%
# Cross validation
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf, X, y, cv=kfold, scoring='accuracy')

# Display the results
print(f"Cross-validation accuracy scores: {cv_scores}")

#%%[markdown]
# Feature Importance:
# By looking at the Feature Importance values, median_gross_income plays a dominant role in determining housing price tiers, while crime rates have a noticeable but secondary impact.
# This analysis highlights that neighborhood income levels are the most crucial factor for classifying housing prices, which aligns with socioeconomic expectations.

# Model Evaluation Interpretation:
# The model achieved an accuracy of 79%, meaning it correctly classified 79% of the neighborhoods into the three housing price tiers (low, medium, high).
# The model performs best for the low and high price tiers (Classes 0 and 2), with slightly lower performance for the medium price tier (Class 1). Overall, the model demonstrates reliable and balanced predictions across the three tiers.

##### XG Boost










# %%
#AJ1--Crime Rate vs Rent Relationship
# Scatter plot for Crime Rate vs Rent
plt.figure(figsize=(10, 6))

# Scatter plot points
plt.scatter(cp_data['total_crimes'], cp_data['price'], color='Blue', alpha=0.7)

# Adding labels and title
plt.title('Crime Rate vs Rent Relationship', fontsize=16)
plt.xlabel('Total Crimes', fontsize=12)
plt.ylabel('Average Rent', fontsize=12)

# Adding gridlines
plt.grid(visible=True, linestyle='--', alpha=0.6)

# Adjust layout for clarity
plt.tight_layout()

# Show plot
plt.show()
# %%
#AJ2--further analysis 1:Analyze the density of points in clusters to determine patterns.
# Hexbin Plot for density analysis
plt.figure(figsize=(10, 6))

# Creating a hexbin plot for density
plt.hexbin(cp_data['total_crimes'], cp_data['price'], gridsize=30, cmap='Blues', mincnt=1)

# Adding colorbar for density
cb = plt.colorbar(label='Density')

# Adding labels and title
plt.title('Density of Crime Rate vs Rent Relationship', fontsize=16)
plt.xlabel('Total Crimes', fontsize=12)
plt.ylabel('Average Rent', fontsize=12)

# Adding grid
plt.grid(visible=True, linestyle='--', alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()
# %%
#AJ3--Crime Rate vs Rent Relationship with Trend Line
from scipy.stats import pearsonr

# Scatter plot with a trend line
plt.figure(figsize=(10, 6))

# Plotting the scatter plot
sns.scatterplot(x='total_crimes', y='price', data=cp_data, alpha=0.7, color='blue', label='Data Points')

# Adding a trend line (using numpy polyfit for linear regression)
z = np.polyfit(cp_data['total_crimes'], cp_data['price'], 1)  # 1 for a linear fit
p = np.poly1d(z)
plt.plot(cp_data['total_crimes'], p(cp_data['total_crimes']), color='orange', linestyle='--', label='Trend Line')

# Calculating the correlation coefficient
correlation, _ = pearsonr(cp_data['total_crimes'], cp_data['price'])
plt.text(50, cp_data['price'].max() * 0.9, f"Correlation: {correlation:.2f}", fontsize=12, color='red')

# Adding labels and title
plt.title('Crime Rate vs Rent Relationship with Trend Line', fontsize=16)
plt.xlabel('Total Crimes', fontsize=12)
plt.ylabel('Average Rent', fontsize=12)
plt.legend()
plt.grid(visible=True, linestyle='--', alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()
# %%
#AJ4--Correlation heat map
# Calculate Pearson correlation between crime rate and property value
crime_property_corr = cp_data[['total_crimes', 'price']].corr()

# Visualize the correlation matrix

sns.heatmap(crime_property_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Crime Rate and Property Value')
plt.show()
# %%
# from statsmodels.tsa.seasonal import seasonal_decompose

# # Decompose both crime rate and property value time series
# crime_decomposition = seasonal_decompose(cp_data['total_crimes'], model='additive', period=12)
# property_decomposition = seasonal_decompose(cp_data['price'], model='additive', period=12)

# # Plot the decompositions
# crime_decomposition.plot()
# plt.suptitle('Crime Rate Decomposition')
# plt.show()

# property_decomposition.plot()
# plt.suptitle('Property Value Decomposition')
# plt.show()
# %%


# Regression problem model 2 

# RANDOM FOREST REGRESSOR 

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
