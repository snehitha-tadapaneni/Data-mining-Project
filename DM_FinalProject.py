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
#1. To what extent do neighborhood crime rates correlate with residential property values across Washington DC, using the combined analysis of the Housing Price dataset and open data crime dataset. 
#   
#2. How do changes in violent crime rates influence median house prices in city of Washington DC, controlling for socioeconomic factors, using quarterly crime statistics and residential property sales data
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

#%%[markdown]
## Data Preparation
# We have merged and aggregated both datasets based on `census_tract` and offense counts. 
# The final dataset contains 21 columns, including house characteristics (e.g., price, rooms, bathrooms) and detailed crime-related features.
# Specifically, the crime-related deatures include each offense category (e.g., ARSON, BURGLARY, HOMICIDE, THEFT), 
# method of committing crimes (e.g., GUN, KNIFE), and shift (e.g., DAY, NIGHT).
# This comprehensive dataset enables a thorough analysis of the relationship between housing attributes and crime rates.

# Step 1: We have extracted only the year 2018 from both the datasets - house_price18, crime_18.
# Step 2: We have dropped the unecessary columns from both the datasets(code below).

############################code for dropping columns from house_price18 - sayam



###########################code for dropping columns from crime18 - snehitha
# Reading the crime dataset (From years 2014 - 2018)
crime_2014_2018 = pd.read_csv('Crime_Incidents_in_2014_to_2018.csv')

# looking at the dataset
crime_2014_2018.head

# Dropping a few columns that are irrelevant to our analysis
columns_to_drop = ['REPORT_DAT', 'WARD', 'ANC', 'NEIGHBORHOOD_CLUSTER', 'BLOCK_GROUP', 'BID', 'DISTRICT', 'PSA', 'X', 'Y', 'CCN', 'XBLOCK', 'YBLOCK', 'VOTING_PRECINCT', 'LATITUDE', 'LONGITUDE', 'END_DATE', 'OBJECTID']  # Replace with the actual column names you want to drop
crime_2014_2018 = crime_2014_2018.drop(columns=columns_to_drop)

# Filtering rows where the extracted year is '2018'
crime_2014_2018['Year_Extracted'] = crime_2014_2018['START_DATE'].astype(str).str[:4]
crime_18 = crime_2014_2018[crime_2014_2018['Year_Extracted'] == '2018']

# Displaying the filtered dataset
print(crime_18.head())
crime_18.info()

# Saving the Crime Incident dataset extracted with 2018 year into crime18.csv file
crime_18.to_csv('crime18.csv')

#%%[markdown]
# Step 3: We have aggregated the crime dataset based on the `census_tract` and categorized them by three key features: `offense`, `method`, and `shift`
# For each feature, we counted the number of incidents per census tract.

#%%
# Loading the Crime Incident dataset in 2018 
crime18 = pd.read_csv('crime18.csv')

# Converting column names into lowercase
crime18.columns = crime18.columns.str.lower()
# Converting 'census_tract' to string type
crime18['census_tract'] = crime18['census_tract'].astype(str)

########## Aggregation for All Features ##########

# Aggregating crime incidents by census tract and offense, counting incidents
crime_counts_by_offense = crime18.groupby(['census_tract','offense'])['unnamed: 0'].count().reset_index()
crime_counts_by_offense = crime_counts_by_offense.rename(columns={'unnamed: 0': 'count'})

# Aggregating crime incidents by census tract and method, counting incidents
crime_counts_by_method = crime18.groupby(['census_tract','method'])['unnamed: 0'].count().reset_index()
crime_counts_by_method = crime_counts_by_method.rename(columns={'unnamed: 0': 'count'})

# Aggregating crime incidents by census tract and shift, counting incidents
crime_counts_by_shift = crime18.groupby(['census_tract','shift'])['unnamed: 0'].count().reset_index()
crime_counts_by_shift = crime_counts_by_shift.rename(columns={'unnamed: 0': 'count'})

#%%[markdown]
# Step 4: We have pivoted the aggregated data to create separate tables for offenses, methods, and shifts, with census_tract as the index.

#%%
########## Pivot for All Features ##########

# Pivoting aggregated data for offenses by census tract
crime_pivot_offense = crime_counts_by_offense.pivot(index='census_tract',
                                                    columns='offense',
                                                    values='count')

# Pivoting aggregated data for methods by census tract
crime_pivot_method = crime_counts_by_method.pivot(index='census_tract',
                                                  columns='method',
                                                  values='count')

# Pivoting aggregated data for shifts by census tract
crime_pivot_shift = crime_counts_by_shift.pivot(index='census_tract',
                                                columns='shift',
                                                values='count')
#%%[markdown]
# Step 5: We have concatenated the pivot tables for offenses, methods, and shifts into a single combined dataset, aligning them by census_tract.
#%%
########## Combine All Pivot Tables ##########

# Concatenating offense, method, and shift pivot tables along columns
crime_pivot_combined = pd.concat([crime_pivot_offense, 
                                  crime_pivot_method, 
                                  crime_pivot_shift
                                  ], axis = 1)

# Reseting the index to convert 'census_tract' from index to a column
crime_pivot_combined = crime_pivot_combined.reset_index()

# Converting 'census_tract' to string type
crime_pivot_combined['census_tract'] = crime_pivot_combined['census_tract'].astype('str')
#%%[markdown]
# Step 6: We merged the housing data in 2018 with combined crime counts pivot table for features (offense, method, and shift) by census tract
#%%
########## Merge Housing Table with Crime Table ##########

# Loading housing data in 2018 
# Read the text file with space as a delimiter
House_price18 = pd.read_csv('House_price18')

# Save the DataFrame to CSV
House_price18.to_csv('House_price18.csv', index=False)

housing18 = pd.read_csv('House_price18.csv') 

# Converting column names into lowercase
housing18.columns = housing18.columns.str.lower()
# Converting 'census_tract' to string type
housing18['census_tract'] = housing18['census_tract'].astype('str')

# Merging housing table in 2018 with census tract level crime counts by `offense`, `method`, and `shift`
df_combined = housing18.merge(crime_pivot_combined, 
                                left_on=['census_tract'], 
                                right_on=['census_tract'])
#%%
df_combined
df_combined.info()
df_combined.to_csv('final_data18.csv') 

# Now, we have our final dataset saved as final_data18.csv. Let's proceed with our Exploration!
## Data Exploring and Cleaning
#%%
# Reading the dataset into cp_data
cp_data = pd.read_csv("final_data18.csv")

# %%
# Print first 5 rows of the dataset
cp_data.head()

# Print last 5 rows of the dataset
cp_data.tail()

#%%
# shape of our merged dataset
cp_data.shape

# Checking columns
cp_data.columns

#%%
# Statistics of the data
cp_data.describe()

# Checking the datatypes
cp_data.info()

# Checking for null values
cp_data.isnull().sum()
# These missing values show us where there is no crime rate
sns.heatmap(cp_data.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values in Dataset")
plt.show()

#############################sayam code to handle missing values and dropping few columns#############################


# Now, that we have cleaned our dataset. Let's Explore and learn more about our features.
## Data Visualization: Univariate Analysis
# %%
# Histograms for numerical features
num_cols = ['bathrm', 'rooms', 'kitchens', 'fireplaces', 'ARSON', 'ASSAULT W/DANGEROUS WEAPON', 'BURGLARY', 'HOMICIDE', 'MOTOR VEHICLE THEFT', 'ROBBERY', 'SEX ABUSE', 'THEFT F/AUTO', 'THEFT/OTHER', 'GUN', 'KNIFE', 'OTHERS', 'DAY', 'EVENING', 'MIDNIGHT']
cp_data[num_cols].hist(figsize=(10, 8), layout=(6, 4 ), edgecolor='black')
plt.suptitle('Distributions of Numerical Features')
plt.show()


# analysing the target variable - plotting a distribution to understand price
plt.figure(figsize=(10, 6))
sns.histplot(cp_data['price'], kde=True)
plt.title("Distribution of Housing Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# %%
# As the price distribution is highly skewed, lets look at the outliers
# Boxplot for detecting outliers in price
plt.figure(figsize=(10, 6))
sns.boxplot(x=cp_data['price'])
plt.title("Boxplot of Housing Prices")
plt.show()

# Removing outliers would not be appropriate for our analysis, as we can draw insights from higher hosuing prices




## Bivariate Analysis
# %%
# Heatmap to understand relationship bw price and other variables
# Select only numerical columns
numerical_cols = cp_data.select_dtypes(include=['float64', 'int64']).columns
numerical_df = cp_data[numerical_cols]

# Compute the correlation matrix
corr = numerical_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title('Correlation Heatmap')
plt.show()


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

# %%
# Adding total crime columns
cp_data['total_crimes'] = cp_data[['ARSON', 'ASSAULT W/DANGEROUS WEAPON', 'BURGLARY', 'HOMICIDE', 
                             'MOTOR VEHICLE THEFT', 'ROBBERY', 'SEX ABUSE', 
                             'THEFT F/AUTO', 'THEFT/OTHER']].sum(axis=1)

# Define crime categories
violent_crimes = ['ARSON', 'HOMICIDE', 'ASSAULT W/DANGEROUS WEAPON', 'ROBBERY', 'SEX ABUSE', 'GUN']
non_violent_crimes = ['THEFT F/AUTO', 'BURGLARY', 'MOTOR VEHICLE THEFT', 'OTHERS']

# Calculate sums
cp_data['violent_crimes'] = cp_data[violent_crimes].sum(axis=1)
cp_data['non_violent_crimes'] = cp_data[non_violent_crimes].sum(axis=1)

sns.pairplot(cp_data[['price', 'rooms', 'bathrm', 'kitchens', 'fireplaces', 'total_crimes', 'violent_crimes']])
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
