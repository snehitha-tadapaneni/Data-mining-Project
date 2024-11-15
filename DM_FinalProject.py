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
# We have merged and aggregated both datasets based on `census_tract` and offense counts. The final dataset contains 21 columns, including house characteristics (e.g., price, rooms, bathrooms) and offense categories (e.g., ARSON, BURGLARY, HOMICIDE, THEFT). This comprehensive dataset enables a thorough analysis of the relationship between housing attributes and crime rates.

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

# Step 3: We have aggregated the crime dataset based on the census tract and count of offenses.(code below)

########################code for merging and aggregate functions - annie




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

# we have no missing values in our dataset
