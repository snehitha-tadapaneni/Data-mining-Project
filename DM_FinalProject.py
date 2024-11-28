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
# The final dataset contains 21 columns, including house characteristics (e.g., price, rooms, bathrooms) and detailed crime-related features.
# Specifically, the crime-related deatures include each offense category (e.g., ARSON, BURGLARY, HOMICIDE, THEFT), 
# method of committing crimes (e.g., GUN, KNIFE), and shift (e.g., DAY, NIGHT).
# This comprehensive dataset enables a thorough analysis of the relationship between housing attributes and crime rates.

# Step 1: We have extracted year from 2014 to 2018 from both the datasets - crime and housing prices.
# Step 2: We have dropped the unecessary columns from both the datasets.

# Reading the crime dataset (From years 2014 - 2018)
crime_2014_2018 = pd.read_csv('Crime_Incidents_in_2014_to_2018.csv')

# looking at the dataset
# crime_2014_2018.head

# Dropping a few columns that are irrelevant to our analysis
columns_to_drop = ['REPORT_DAT', 'WARD', 'ANC', 'NEIGHBORHOOD_CLUSTER', 'BLOCK_GROUP', 'BID', 'DISTRICT', 'PSA', 'X', 'Y', 'CCN', 'XBLOCK', 'YBLOCK', 'VOTING_PRECINCT', 'LATITUDE', 'LONGITUDE', 'END_DATE', 'OBJECTID', 'Unnamed: 24', 'Unnamed: 25']  # Replace with the actual column names you want to drop
crime_2014_2018 = crime_2014_2018.drop(columns=columns_to_drop)

# Filtering rows where year is between 2014 and 2018
crime_2014_2018['year'] = crime_2014_2018['START_DATE'].astype(str).str[:4]
crime_2014_2018 = crime_2014_2018[crime_2014_2018['year'].isin(['2014', '2015', '2016', '2017', '2018'])]

# Displaying the filtered dataset
# crime_2014_2018.info()

#%%[markdown]
# Step 3: Aggregate annual crime statistics grouped by `offense`, `method` and `shift`

#%%
# Convert column names into lowercases
crime_2014_2018.columns = crime_2014_2018.columns.str.lower()

# Check for missing values in each column
crime_2014_2018.isnull().sum()

# Drop rows with missing values in `year` and `census_tract`
crime_2014_2018 = crime_2014_2018.dropna(subset=['year', 'census_tract'])

# Convert `census_tract` and `start_year` into integer data types
crime_2014_2018['census_tract'] = pd.to_numeric(crime_2014_2018['census_tract'], errors='coerce').astype(int)
crime_2014_2018['year'] = crime_2014_2018['year'].astype(int)

# Apply one-hot encoding to offense, method, and shift columns to prepare for aggregation
crime_dummies = pd.get_dummies(crime_2014_2018[['year', 'census_tract', 'offense', 'method', 'shift']], columns=['offense', 'method','shift'])

# Aggregate crime data by year and census tract, summing counts across offense, method, and shift categories
crime_grouped = crime_dummies.groupby(['year', 'census_tract']).sum().reset_index()

# Filter the aggregated dataset to include only data from 2014 onwards
crime_grouped = crime_grouped[crime_grouped['year'] >= 2014]

# Display info about the grouped dataset
# crime_grouped.info()

#%%[markdown]
# Step 4: We merged the housing data with annual crime statistics

# Loading housing data in 2018 
dc_housing = pd.read_csv("dc_house_price.csv")
dc_housing.columns =  dc_housing.columns.str.lower()

# Drop irrelevant columns
drop_cols = ['price.1', 'source']
dc_housing.drop(drop_cols, axis = 1, inplace = True)

# Convert `census_tract` and `year` columns into integer data type
dc_housing['census_tract'] = pd.to_numeric(dc_housing['census_tract'], errors='coerce').astype(int)
dc_housing['year'] = dc_housing['year'].astype(int)

# Merging housing table in 2018 with census tract level crime counts by `offense`, `method`, and `shift`
df_combined = dc_housing.merge(crime_grouped, 
                            left_on=['year', 'census_tract'], 
                            right_on=['year','census_tract'],
                            how = 'inner')

df_combined.info()
#%%
df_combined.to_csv('final_return.csv') 

#%%[markdown]
# Now, we have our final dataset saved as final_return.csv. Let's proceed with our Exploration!
## Data Exploring and Cleaning
#%%
# Reading the dataset into cp_data
cp_data = pd.read_csv("final_return.csv")

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

# Checking for null values/ missing values
cp_data.isnull().sum()
# A heatmap to visualise the missing data points if any
sns.heatmap(cp_data.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values in Dataset")
plt.show()
#%%[markdown]
# We do not have any missing values in our dataset. Phew, no need to handle them!

# Renaming the columns, all to upper cases
cp_data.columns = cp_data.columns.str.upper()

# Now, that we have cleaned our dataset. Let's Explore and learn more about our features.

#

## Data Visualization: Univariate Analysis
# <br>
# Distribution for all numerical features and the target variable
# %%
# Histograms for numerical features
num_cols = ['BATHRM', 'ROOMS', 'KITCHENS', 'FIREPLACES', 'OFFENSE_ARSON', 'OFFENSE_ASSAULT W/DANGEROUS WEAPON', 'OFFENSE_BURGLARY', 'OFFENSE_HOMICIDE', 'OFFENSE_MOTOR VEHICLE THEFT', 'OFFENSE_ROBBERY', 'OFFENSE_SEX ABUSE', 'OFFENSE_THEFT F/AUTO', 'OFFENSE_THEFT/OTHER', 'METHOD_GUN', 'METHOD_KNIFE', 'METHOD_OTHERS', 'SHIFT_DAY', 'SHIFT_EVENING', 'SHIFT_MIDNIGHT']
cp_data[num_cols].hist(figsize=(10, 8), layout=(6, 4 ), edgecolor='black')
plt.suptitle('Distributions of Numerical Features')
plt.show()

# Analysing the target variable - plotting a distribution to understand price
plt.figure(figsize=(10, 6))
sns.histplot(cp_data['PRICE'], kde=True)
plt.title("Distribution of Housing Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# %%[markdown]
# As the price distribution is highly skewed, lets look at the outliers in the target variable.
# Boxplot for detecting outliers in price
plt.figure(figsize=(10, 6))
sns.boxplot(x=cp_data['PRICE'])
plt.title("Boxplot of Housing Prices")
plt.show()

# As we can see, there are price values too high that can impact the models ability to understand and interpret the the data. To make our analysis, more specific and less comples, we are sticking to certain pricing values.
# We will only interpret residential property values untill 1500000$. Let's consider all the other points as outliers and remove them.

#%%
# Removing the outliers from the target variable: price
cp_data = cp_data[cp_data['PRICE']<1500000]

#%%
# Frequency of each method type: Plot for price vs method types
methods = ['METHOD_GUN', 'METHOD_KNIFE', 'METHOD_OTHERS']
method_sums = cp_data[methods].sum()

plt.figure(figsize=(8, 6))
method_sums.plot(kind='bar', color=['skyblue', 'orange', 'green'], alpha=0.8)
plt.title('Frequency of Each Method Type', fontsize=14)
plt.xlabel('Method Type', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=0)
plt.show()

# Distribution of price after removing the outliers
plt.figure(figsize=(8, 6))
sns.histplot(cp_data['PRICE'], kde=True, color='purple', bins=30)
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
numerical_cols = cp_data.select_dtypes(include=['float64', 'int64']).columns
numerical_df = cp_data[numerical_cols]

# Compute the correlation matrix
corr = numerical_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title('Correlation Heatmap')
plt.show()
#%%[markdown]
# *Price Correlations Obervations*:
# <br> 1. The variable PRICE has a moderate positive correlation with CENSUS_TRACT (correlation ~0.54), showing some geographical influence on prices.
# <br> 2. BATHRM, ROOMS, and FIREPLACES show mild positive correlations with PRICE, indicating that these features might drive higher property values.
# <br> 3.  METHOD_GUN and other offense-related variables have a negative correlation with PRICE, implying that crime rates might negatively impact property values.
# <br><br>
# Let us dig deep to support our observations.

# Plot for Price vs rooms, bathrooms, kitchens
#%%
# Prices vs rooms
plt.figure(figsize=(10, 6))
sns.boxplot(x='rooms', y='price', data=cp_data)
plt.title("Housing Prices Based on Number of Rooms")
plt.xlabel("Number of Rooms")
plt.ylabel("Price")
plt.show()

# %%
# Prices vs bathrooms
plt.figure(figsize=(10, 6))
sns.boxplot(x='bathrm', y='price', data=cp_data)
plt.title("Housing Prices Based on Number of Bathrooms")
plt.xlabel("Number of Bathrooms")
plt.ylabel("Price")
plt.show()

# %%
# Prices vs kitchens
plt.figure(figsize=(10, 6))
sns.boxplot(x='kitchens', y='price', data=cp_data)
plt.title("Housing Prices Based on Number of Kitchens")
plt.xlabel("Number of Kitchens")
plt.ylabel("Price")
plt.show()

#%%[markdown]
# # Scatter plot: All method types(gun, knife, others) vs housing prices
#%%
# Prepare data for scatter plot
plt.figure(figsize=(10, 6))

# Plot for METHOD_GUN
plt.scatter(
    cp_data.loc[cp_data['METHOD_GUN'] == 1, 'PRICE'],
    cp_data.loc[cp_data['METHOD_GUN'] == 1].index,
    color='red', label='Gun', alpha=0.6
)

# Plot for METHOD_KNIFE
plt.scatter(
    cp_data.loc[cp_data['METHOD_KNIFE'] == 1, 'PRICE'],
    cp_data.loc[cp_data['METHOD_KNIFE'] == 1].index,
    color='blue', label='Knife', alpha=0.6
)

# Plot for METHOD_OTHERS
plt.scatter(
    cp_data.loc[cp_data['METHOD_OTHERS'] == 1, 'PRICE'],
    cp_data.loc[cp_data['METHOD_OTHERS'] == 1].index,
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
corr_gun, p_gun = spearmanr(cp_data['PRICE'], cp_data['METHOD_GUN'])
corr_knife, p_knife = spearmanr(cp_data['PRICE'], cp_data['METHOD_KNIFE'])
corr_others, p_others = spearmanr(cp_data['PRICE'], cp_data['METHOD_OTHERS'])

# Display the results
print(f"Spearman Correlation for method_GUN: {corr_gun}, p-value: {p_gun}")
print(f"Spearman Correlation for method_KNIFE: {corr_knife}, p-value: {p_knife}")
print(f"Spearman Correlation for method_OTHERS: {corr_others}, p-value: {p_others}")

#%%[markdown]
# Let us state our hypothesis,<br>
# Null Hypothesis (H₀): There is no monotonic relationship between price and method_GUN, method_KNIFE, method_OTHERS.
# Alternative Hypothesis (H₁): There is a significant monotonic relationship between price and method_GUN, method_KNIFE, method_OTHERS.

# 1. For method_GUN:<br>
# Interpretation: Since the p-value is 0.0 (which is less than 0.05), we reject the null hypothesis, indicating a significant monotonic relationship between price and method_GUN.

# 2. For method_KNIFE:<br>
# Interpretation: The p-value is also 0.0, which is less than 0.05, so we reject the null hypothesis, indicating a significant monotonic relationship between price and method_KNIFE.

# 3. For method_OTHERS:<br>
# Interpretation: Since the p-value is 0.976 (which is greater than 0.05), we fail to reject the null hypothesis, indicating no significant monotonic relationship between price and method_OTHERS.
#<br>

# Pair plot between crime categories vs the price
# %%
# Define crime categories as violent and property crimes and encode them by yes(1) and no(0)
cp_data['VIOLENT_CRIMES'] = (cp_data['OFFENSE_ASSAULT W/DANGEROUS WEAPON'] +
                       cp_data['OFFENSE_HOMICIDE'] +
                       cp_data['OFFENSE_ROBBERY'] +
                       cp_data['OFFENSE_SEX ABUSE']).apply(lambda x: 1 if x > 0 else 0)

cp_data['PROPERTY_CRIMES'] = (cp_data['OFFENSE_ARSON'] +
                        cp_data['OFFENSE_BURGLARY'] +
                        cp_data['OFFENSE_MOTOR VEHICLE THEFT'] +
                        cp_data['OFFENSE_THEFT F/AUTO'] +
                        cp_data['OFFENSE_THEFT/OTHER']).apply(lambda x: 1 if x > 0 else 0)

# Visualize relationships using pairplot
sns.pairplot(cp_data[['PRICE', 'ROOMS', 'BATHRM', 'KITCHENS', 'FIREPLACES','VIOLENT_CRIMES', 'PROPERTY_CRIMES']])
plt.show()





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
