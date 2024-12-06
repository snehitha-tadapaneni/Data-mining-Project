# Data-mining-Project
## Course - DATS_6103_Introduction_to_Data_Mining

### TOPIC: Analysing the impact of Neighborhood crime rates on residential property in Washington DC between the years 2014-2018

- To what extent do neighborhood crime rates correlate with residential property values across Washington DC between 2014-2018, using the combined analysis of the Housing Price dataset and opendata crime dataset.<br>
- Presenting the understanding through two SMART questions which shed light on the regression and classification problems with regards to data and how much to socio-economic factors and crime factors affect the price of a property our aim is to uncover this underlying relationship.<br>
- How do changes in violent crime rates influence median house prices in city of Washington DC during 2020-2024, controlling for socioeconomic factors, using quarterly crime statistics and residential property sales data

## Dependancies to get started with the dataset and preprocessing 

```bash
pip install numpy
pip isntall matplotlib
pip install pandas
pip install sklearn
```
## Below is the workflow for getting the dataset with no outliers and null values 
```bash
# Load the csv file 
df = read_csv('final_return_new.csv')
```
```bash
# Heatmap to visualize the correlation
```
```bash
# Code to use IQR (Interquartile Range) to remove outliers and plot boxplot and histogram to visualize the price column 
def iqr(df):
    df1 = df.copy()
    
    q1, q3 = np.percentile(df['price'], 25), np.percentile(df['price'], 75)
    
    iqr = q3-q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    df1 = df1[(df1['price'] >= lower) & (df1['price'] <= upper)]
    
    #Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df1['price'])
    plt.title("Boxplot of Housing Prices")
    plt.show()

    # Histplot
    plt.figure(figsize=(10, 6))
    sns.histplot(df1['price'], kde=True)
    plt.title("Distribution of Housing Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.show()
```
Post Outlier removal:
- Boxlpot
- Histogram
<img width="799" alt="Screenshot 2024-12-05 at 10 38 40 PM" src="https://github.com/user-attachments/assets/848b3943-c24a-4f66-9886-a92e82e62b20">

<img width="657" alt="Screenshot 2024-12-05 at 10 46 00 PM" src="https://github.com/user-attachments/assets/84a68272-a75a-4465-8027-040094b87116">


Dataset :- 

House Pricing Dataset:  https://www.kaggle.com/datasets/christophercorrea/dc-residential-properties?select=raw_address_points.csv
the house_price18.csv dataset is the cleaned dataset

DC Crime Rate: https://opendata.dc.gov/datasets/c5a9f33ffca546babbd91de1969e742d_6/explore?location=38.903935%2C-77.012050%2C10.79

Merged dataset is named as final_return_new.csv
Link to the data : https://drive.google.com/file/d/1i8ANBUP_x9Mtvbmnt3jOKkgVc248LhvS/view?usp=sharing



