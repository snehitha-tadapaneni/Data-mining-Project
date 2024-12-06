# Data-mining-Project
## Course - DATS_6103_Introduction_to_Data_Mining

### TOPIC: Analysing the impact of Neighborhood crime rates on residential property in Washington DC between the years 2014-2018

- To what extent do neighborhood crime rates correlate with residential property values across Washington DC between 2014-2018, using the combined analysis of the Housing Price dataset and opendata crime dataset.<br>
- Presenting the understanding through two SMART questions which shed light on the regression and classification problems with regards to data and how much to socio-economic factors and crime factors affect the price of a property our aim is to uncover this underlying relationship.<br>
- How do changes in violent crime rates influence median house prices in city of Washington DC during 2020-2024, controlling for socioeconomic factors, using quarterly crime statistics and residential property sales data

## Worked on two SMART questions which focus on the following topics:
- regression predicting the price of the property based on the various criime realted and socio-economic featues
- classificaion based problem to classify the prices

## Creating an enviornment 
For Mac/Linux
```bash
# Create virtual environment 
python -m venv myenv

# Activate virtual environment
source myenv/bin/activate

# Deactivate when done
deactivate
```
For windows
```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
myenv\Scripts\activate

# Deactivate when done
deactivate
```

## Dependancies to get started with the dataset and preprocessing 
```bash
pip install -r requirements.txt
```
## Below is the workflow for getting the dataset with no outliers and null values 
```bash
# Load the csv file 
df = read_csv('final_return_new.csv')
```
```bash
# Heatmap to visualize the correlation
# Takes in only numerical columns and creates a heatmap to understand the correlation between variables
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
```
Correaation between numerical variables 
![Screenshot 2024-12-06 at 3 21 05 PM](https://github.com/user-attachments/assets/0e2de543-ba4e-4387-ba26-28749883bcc9)


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

Above steps help any user to begin with the project get teh datset remove null and outlier values <br>
Below is the list of models used the user can go ahead and download the specific libraries and dependancies then run the models to get similar output or fine tune to work on custom data

## Regression Models:
- LightGradient Boosting
- Random Forest Regressor

## Classification Models
- Random Forest Classifier
- XG Bossting

## Dataset :- 

House Pricing Dataset:  https://www.kaggle.com/datasets/christophercorrea/dc-residential-properties?select=raw_address_points.csv
the house_price18.csv dataset is the cleaned dataset

DC Crime Rate: https://opendata.dc.gov/datasets/c5a9f33ffca546babbd91de1969e742d_6/explore?location=38.903935%2C-77.012050%2C10.79

Merged dataset is named as final_return_new.csv
Link to the data : https://drive.google.com/file/d/1i8ANBUP_x9Mtvbmnt3jOKkgVc248LhvS/view?usp=sharing



