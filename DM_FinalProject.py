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
plt.figure(figsize=(10, 6))
sns.histplot(cp_data['price'], kde=True)
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
# Heatmap to understand relationship bw price and other variables
# Selected only numerical columns
numerical_cols = cp_data_cleaned.select_dtypes(include=['float64', 'int64']).columns
numerical_df = cp_data_cleaned[numerical_cols]

# Compute the correlation matrix
corr = numerical_df.corr()

# Plot the heatmap
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

# Plot for Price vs rooms, bathrooms, kitchens
#%%
# Prices vs rooms
plt.figure(figsize=(10, 6))
sns.boxplot(x='rooms', y='price', data=cp_data_cleaned)
plt.title("Housing Prices Based on Number of Rooms")
plt.xlabel("Number of Rooms")
plt.ylabel("Price")
plt.show()

# %%
# Prices vs bathrooms
plt.figure(figsize=(10, 6))
sns.boxplot(x='bathrm', y='price', data=cp_data_cleaned)
plt.title("Housing Prices Based on Number of Bathrooms")
plt.xlabel("Number of Bathrooms")
plt.ylabel("Price")
plt.show()


#%%
# Prices vs bed room
plt.figure(figsize=(10, 6))
sns.boxplot(x='bedrm', y='price', data=cp_data_cleaned)
plt.title("Housing Prices Based on Number of Bed Rooms")
plt.xlabel("Number of Rooms")
plt.ylabel("Price")
plt.show()

#%%
# Prices vs ward
plt.figure(figsize=(10, 6))
sns.barplot(x='ward', y='price', data=cp_data_cleaned)
plt.title("Housing Prices Based on Number of ward")
plt.xlabel("Ward")
plt.ylabel("Price")
plt.show()


#%%[markdown]
# # Scatter plot: All method types(gun, knife, others) vs housing prices
#%%
# Prepare data for scatter plot
plt.figure(figsize=(10, 6))

# Plot for METHOD_GUN
plt.scatter(
    cp_data_cleaned.loc[cp_data_cleaned['method_gun'] == 1, 'price'],
    cp_data_cleaned.loc[cp_data_cleaned['method_gun'] == 1].index,
    color='red', label='Gun', alpha=0.6
)

# Plot for METHOD_KNIFE
plt.scatter(
    cp_data_cleaned.loc[cp_data_cleaned['method_knife'] == 1, 'price'],
    cp_data_cleaned.loc[cp_data_cleaned['method_knife'] == 1].index,
    color='blue', label='Knife', alpha=0.6
)

# Plot for METHOD_OTHERS
plt.scatter(
    cp_data_cleaned.loc[cp_data_cleaned['method_others'] == 1, 'price'],
    cp_data_cleaned.loc[cp_data_cleaned['method_others'] == 1].index,
    color='yellow', label='Others', alpha=0.6
)

# Scatter plot: method types vs price
plt.title('Scatter Plot: Price vs Method Types', fontsize=14)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Index (Observations)', fontsize=12)
plt.legend(title='Method Type')
plt.grid(alpha=0.5, linestyle='--')
plt.show()

#%%[markdown]
# From the above scatter plot, we can see that there are no Method_Others influencing the price values. 
# Let us perform a statistical test(Spearman Correlation) to check the relationship between the price and the method types and prove our point.
#%%
# Calculate Spearman correlation between 'price' and each 'method' type
corr_gun, p_gun = spearmanr(cp_data_cleaned['price'], cp_data_cleaned['method_gun'])
corr_knife, p_knife = spearmanr(cp_data_cleaned['price'], cp_data_cleaned['method_knife'])
corr_others, p_others = spearmanr(cp_data_cleaned['price'], cp_data_cleaned['method_others'])

# Display the results
print(f"Spearman Correlation for method_GUN: {corr_gun}, p-value: {p_gun}")
print(f"Spearman Correlation for method_KNIFE: {corr_knife}, p-value: {p_knife}")
print(f"Spearman Correlation for method_OTHERS: {corr_others}, p-value: {p_others}")

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
plt.figure(figsize=(10, 6))
plt.scatter(cp_data_cleaned['violent_crime_count'], cp_data_cleaned['price'], color='red', alpha=0.6, label='Violent Crimes')
plt.scatter(cp_data_cleaned['property_crime_count'], cp_data_cleaned['price'], color='blue', alpha=0.6, label='Property Crimes')
plt.title('Scatter Plot: Violent and Property Crimes vs Price')
plt.xlabel('Crime Count')
plt.ylabel('Price')
plt.legend()
plt.grid(alpha=0.5)
plt.show()

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
plt.figure(figsize=(10, 6))
plt.scatter(tract_data['violent_crime_count'], tract_data['price'], color='red', alpha=0.6, label='Violent Crimes')
plt.scatter(tract_data['property_crime_count'], tract_data['price'], color='blue', alpha=0.6, label='Property Crimes')
plt.title('Crime Counts vs Average Price by Census Tract')
plt.xlabel('Aggregated Crime Count')
plt.ylabel('Average Price ($)')
plt.legend()
plt.grid(alpha=0.5)
plt.show()

#%%
# Compute correlation coefficients: Price vs violent crime vs property crime
correlation_matrix = cp_data_cleaned[['price', 'violent_crime_count', 'property_crime_count']].corr()

# Display correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Price vs Crime Counts')
plt.show()
#%%[markdown]
# From the above heatmap between Violent Crime, Prperty Crime and Price values, we can say:<br>
# 1. Violent crime has a stronger and negative impact on property prices compared to property crimes.<br>
# 2. Property crimes are weakly related to prices, indicating they may not be a strong factor influencing property values in the dataset.<br>
# 3. The moderate positive correlation between violent and property crimes suggests that crime types are somewhat related in occurrence.<br>

#%%
cp_data_cleaned['ward'] = cp_data_cleaned['ward'].astype(str)

#%%
# Comparing prices based on the crimes in each ward
grouped_data = cp_data_cleaned.groupby('ward').agg({
    'price': 'median',
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
#melted_crime_data = grouped_data.melt(
#    id_vars=['ward'], value_vars=['violent_crime_count', 'property_crime_count'],
#    var_name='Crime Type', value_name='Crime Count'
#)
#sns.barplot(data=melted_crime_data, x='ward', y='Crime Count', hue='Crime Type', ax=axes[1], palette='viridis')
#axes[1].set_title('Ward vs Crime Counts')
#axes[1].set_xlabel('Ward')
#axes[1].set_ylabel('Total Crime Counts')
#axes[1].legend(title='Crime Type')

# Adjust layout
plt.tight_layout()
plt.show()






########YOUR VISUALIZATIONS AND TESTING HERE################




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
