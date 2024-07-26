import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import chart_studio
import chart_studio.plotly as py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
import scipy.stats as stats
import os
import chart_studio.tools

chart_studio.tools.set_credentials_file(username='bikerdouglas', api_key='rGRSQ7yA1eqDryDgGEEl')

world = pd.read_csv('/Users/michaelphillipacosta/dai/projects/religion_crimes_2/data/world_data.csv')

rel_restricts = pd.read_stata('/Users/michaelphillipacosta/dai/projects/religion_crimes_2/data/relig_restrict_2007to2016.dta')

countryScope = {
    "AUT": ["Austria"],
    "ALB": ["Albania"],
    "AND": ["Andorra"],
    "AFG": ["Afghanistan"],
    "AGO": ["Angola"],
    "AZE": ["Azerbaijan"],
    "ARG": ["Argentina"],
    "DZA": ["Algeria"],
    "ATG": ["Antigua and Barbuda"],
    "BHS": ["Bahamas"],
    "BRB": ["Barbados"],
    "ARM": ["Armenia"],
    "BLZ": ["Belize"],
    "BHR": ["Bahrain"],
    "BEL": ["Belgium"],
    "AUS": ["Australia"],
    "BEN": ["Benin"],
    "BRN": ["Brunei"],
    "BWA": ["Botswana"],
    "BGD": ["Bangladesh"],
    "BTN": ["Bhutan"],
    "BGR": ["Bulgaria"],
    "KHM": ["Cambodia"],
    "BLR": ["Belarus"],
    "BOL": ["Bolivia"],
    "BFA": ["Burkina Faso"],
    "CAN": ["Canada"],
    "BIH": ["Bosnia-Herzegovina", "Bosnia and Herzegovina"],
    "CAF": ["Central African Republic"],
    "BDI": ["Burundi"],
    "CHN": ["China", "PRC China"],
    "BRA": ["Brazil"],
    "TCD": ["Chad"],
    "COG": ["Republic of the Congo", "Congo"],
    "COM": ["Comoros"],
    "CMR": ["Cameroon"],
    "DNK": ["Denmark"],
    "COD": ["Democratic Republic of the Congo"],
    "CUB": ["Cuba"],
    "CPV": ["Cape Verde"],
    "DJI": ["Djibouti"],
    "CRI": ["Costa Rica"],
    "CZE": ["Czech Republic", "Czechia"],
    "CHL": ["Chile"],
    "DMA": ["Dominica"],
    "CIV": ["Ivory Coast"],
    "ECU": ["Ecuador"],
    "COL": ["Colombia", "Colombia (Medellin)"],
    "DOM": ["Dominican Republic"],
    "ERI": ["Eritrea"],
    "SLV": ["El Salvador"],
    "HRV": ["Croatia"],
    "GAB": ["Gabon"],
    "EST": ["Estonia"],
    "FJI": ["Fiji"],
    "CYP": ["Cyprus"],
    "GMB": ["Gambia"],
    "ETH": ["Ethiopia"],
    "FRA": ["France"],
    "EGY": ["Egypt"],
    "GEO": ["Georgia"],
    "GRD": ["Grenada"],
    "GHA": ["Ghana"],
    "GNQ": ["Equatorial Guinea"],
    "HTI": ["Haiti"],
    "IRQ": ["Iraq"],
    "GRC": ["Greece"],
    "FIN": ["Finland"],
    "HND": ["Honduras"],
    "IRL": ["Ireland"],
    "GTM": ["Guatemala"],
    "DEU": ["Germany", "German Federal Republic"],
    "HKG": ["Hong Kong"],
    "ISR": ["Israel"],
    "GNB": ["Guinea-Bissau"],
    "GIN": ["Guinea"],
    "KAZ": ["Kazakhstan"],
    "LBN": ["Lebanon"],
    "ISL": ["Iceland"],
    "GUY": ["Guyana"],
    "KEN": ["Kenya"],
    "LSO": ["Lesotho"],
    "IDN": ["Indonesia"],
    "HUN": ["Hungary"],
    "KIR": ["Kiribati"],
    "LBR": ["Liberia"],
    "JAM": ["Jamaica"],
    "IND": ["India"],
    "KSV": ["Kosovo"],
    "LBY": ["Libya"],
    "JOR": ["Jordan"],
    "IRN": ["Iran"],
    "MDG": ["Madagascar"],
    "MRT": ["Mauritania"],
    "KWT": ["Kuwait"],
    "ITA": ["Italy"],
    "MWI": ["Malawi"],
    "MUS": ["Mauritius"],
    "LAO": ["Laos"],
    "JPN": ["Japan"],
    "MYS": ["Malaysia"],
    "MEX": ["Mexico"],
    "LTU": ["Lithuania"],
    "KGZ": ["Kyrgyzstan"],
    "LVA": ["Latvia"],
    "LIE": ["Liechtenstein"],
    "LUX": ["Luxembourg"],
    "MKD": ["Republic of Macedonia", "North Macedonia"],
    "MLI": ["Mali"],
    "MHL": ["Marshall Islands"],
    "MDA": ["Moldova"],
    "MNG": ["Mongolia"],
    "NRU": ["Nauru"],
    "NLD": ["Netherlands", "The Netherlands"],
    "NER": ["Niger"],
    "PSE": ["Palestinian territories", "Palestine"],
    "PNG": ["Papua New Guinea"],
    "KNA": ["St. Kitts and Nevis"],
    "SRB": ["Serbia"],
    "SLE": ["Sierra Leone"],
    "KOR": ["South Korea"],
    "LKA": ["Sri Lanka"],
    "CHE": ["Switzerland"],
    "TWN": ["Taiwan"],
    "TON": ["Tonga"],
    "TUN": ["Tunisia"],
    "ARE": ["United Arab Emirates"],
    "USA": ["United States", "The USA", "USA", "United States of America"],
    "VEN": ["Venezuela"],
    "SSD": ["South Sudan"],
    "MAC": ["Macau", "Macao"],
    "MLT": ["Malta"],
    "FSM": ["Federated States of Micronesia", "Micronesia"],
    "MCO": ["Monaco"],
    "NAM": ["Namibia"],
    "NPL": ["Nepal"],
    "NGA": ["Nigeria"],
    "PLW": ["Palau"],
    "PAN": ["Panama"],
    "PRT": ["Portugal"],
    "QAT": ["Qatar"],
    "LCA": ["St. Lucia"],
    "VCT": ["St. Vincent and the Grenadines"],
    "SEN": ["Senegal"],
    "SYC": ["Seychelles"],
    "SGP": ["Singapore"],
    "ZAF": ["South Africa"],
    "ESP": ["Spain", "Spain (Murcia)"],
    "SYR": ["Syria"],
    "TJK": ["Tajikistan"],
    "TTO": ["Trinidad and Tobago"],
    "TUR": ["Turkey"],
    "UKR": ["Ukraine"],
    "GBR": ["United Kingdom", "Northern Ireland"],
    "VNM": ["Vietnam"],
    "ESH": ["Western Sahara"],
    "MDV": ["Maldives"],
    "MNE": ["Montenegro"],
    "MAR": ["Morocco"],
    "MOZ": ["Mozambique"],
    "MMR": ["Burma (Myanmar)", "Myanmar"],
    "NOR": ["Norway"],
    "OMN": ["Oman"],
    "PAK": ["Pakistan"],
    "ROU": ["Romania"],
    "RUS": ["Russia"],
    "RWA": ["Rwanda"],
    "SVK": ["Slovakia"],
    "SVN": ["Slovenia"],
    "SLB": ["Solomon Islands"],
    "SOM": ["Somalia"],
    "TZA": ["Tanzania"],
    "THA": ["Thailand"],
    "TLS": ["Timor-Leste"],
    "TGO": ["Togo"],
    "URY": ["Uruguay"],
    "UZB": ["Uzbekistan"],
    "VUT": ["Vanuatu"],
    "NZL": ["New Zealand"],
    "NIC": ["Nicaragua"],
    "PRY": ["Paraguay"],
    "PER": ["Peru"],
    "PHL": ["Philippines"],
    "POL": ["Poland"],
    "WSM": ["Samoa"],
    "SMR": ["San Marino"],
    "STP": ["Sao Tome and Principe"],
    "SAU": ["Saudi Arabia"],
    "SDN": ["Sudan"],
    "SUR": ["Suriname"],
    "SWZ": ["Swaziland"],
    "SWE": ["Sweden"],
    "TKM": ["Turkmenistan"],
    "TUV": ["Tuvalu"],
    "UGA": ["Uganda"],
    "YEM": ["Yemen"],
    "ZMB": ["Zambia"],
    "ZWE": ["Zimbabwe"]
    }

def standardizeCountryNames(df, country_col, country_scope):
    standardized_countries = {}
    for iso_code, names in country_scope.items():
        for name in names:
            standardized_countries[name.lower()] = iso_code
    df[country_col] = df[country_col].str.lower().map(standardized_countries).fillna(df[country_col])
    return df

# Standardize country names in both datasets
df2 = standardizeCountryNames(rel_restricts, 'Ctry_EditorialName', countryScope)
df1 = standardizeCountryNames(world, 'Country', countryScope)

# Merge the datasets on standardized country names and Year
merged_df1 = pd.merge(rel_restricts, world, how='inner', left_on=['Ctry_EditorialName', 'Question_Year'], right_on=['Country', 'Year'])

# Impute NaN values with the median of each column
columns_to_impute = ['Kidnappings per 100000 people', 'Robberies per 100000 people',
                     'Thefts per 100000 people', 'Homicides per 100000 people']

for column in columns_to_impute:
    median_value = merged_df1[column].median()
    merged_df1[column].fillna(median_value, inplace=True)

# Verify that there are no more NaN values in the specified columns
nan_counts_after_imputation = merged_df1[columns_to_impute].isna().sum()
nan_counts_after_imputation

merged_df1[['GRI', 'SHI']].ffill(inplace=True)

# Identify religious composition columns
religious_columns = [
    'People practicing Judaism as percent of the population',
    'Buddhists as percent of the total population',
    'People practicing Hinduism as percent of the population',
    'Muslims as percent of the total population',
    'Christians as percent of the total population',
    'Non religious people as percent of the population'
]

# Proportional imputation function
def proportional_imputation(df, columns):
    for index, row in df.iterrows():
        row_sum = row[columns].sum()
        missing_count = row[columns].isna().sum()
        
        if missing_count > 0:
            missing_value = (100 - row_sum) / missing_count
            for col in columns:
                if pd.isna(row[col]):
                    df.at[index, col] = missing_value
    
    return df

# Apply proportional imputation to the religious columns
merged_df = proportional_imputation(merged_df1, religious_columns)

# Impute other missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(merged_df.select_dtypes(include=['float64']))

# Convert the imputed data back to a DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=merged_df1.select_dtypes(include=['float64']).columns)

# Check the number of rows after imputing missing values
print("Number of rows after imputing missing values:", imputed_df.shape[0])

# Retain non-numeric columns and concatenate with imputed data
non_numeric_columns = merged_df.select_dtypes(exclude=['float64'])
imputed_df = pd.concat([non_numeric_columns.reset_index(drop=True), imputed_df.reset_index(drop=True)], axis=1)
imputed_df

# Retain non-numeric columns and concatenate with imputed data
non_numeric_columns = merged_df.select_dtypes(exclude=['float64'])
imputed_df = pd.concat([non_numeric_columns.reset_index(drop=True), imputed_df.reset_index(drop=True)], axis=1)
imputed_df

merged_df_years = merged_df1[(merged_df1['Year'] >=2009) & (merged_df1['Year'] ==2013)]

# Calculate the number of null values for each country across the available indicators
available_indicators =[
    'People practicing Judaism as percent of the population', 
    'Buddhists as percent of the total population', 
    'People practicing Hinduism as percent of the population', 
    'Muslims as percent of the total population', 
    'Christians as percent of the total population', 
    'Non religious people as percent of the population',
    'GRI',
    'SHI',
    'Kidnappings per 100000 people',
    'Robberies per 100000 people',
    'Thefts per 100000 people',
    'Homicides per 100000 people'
]

# Calculate the number of null values for each country across the available indicators
null_counts_per_country_available_indicators = merged_df_years.groupby('Ctry_EditorialName')[available_indicators].apply(lambda x: x.isnull().sum())

# Display the countries with the lowest null values for the available indicators
null_counts_per_country_available_indicators_sorted = null_counts_per_country_available_indicators.sum(axis=1).sort_values()
null_counts_per_country_available_indicators_sorted

average_scores2 = merged_df_years.groupby('Ctry_EditorialName')[['GRI', 'SHI']].mean()
average_scores2

# Normalize the scores to a 1-10 scale
def normalize(series):
    return 1 + 9 * (series - series.min()) / (series.max() - series.min())

normalized_scores2 = average_scores2.apply(normalize)

# Calculate the composite score by averaging the normalized GRI and SHI scores
normalized_scores2['Composite Score'] = normalized_scores2.mean(axis=1)

# Display the normalized and composite scores
normalized_scores2

merged_df_years = pd.merge(merged_df_years, normalized_scores2, how='inner', left_on=['Ctry_EditorialName'], right_on=['Ctry_EditorialName'])

merged_df_years.sample(10)

# Define the columns representing the proportion of each religion in the population
religion_columns = [
    'People practicing Judaism as percent of the population',
    'Buddhists as percent of the total population',
    'People practicing Hinduism as percent of the population',
    'Muslims as percent of the total population',
    'Christians as percent of the total population',
    'Non religious people as percent of the population'
]

# Calculate the Religious Diversity Index (RDI)
def calculate_rdi(row):
    proportions = row[religion_columns].fillna(0) / 100
    rdi = 1 - sum(proportions**2)
    return rdi

merged_df_years['RDI'] = merged_df_years.apply(calculate_rdi, axis=1)

# Average RDI for each country
average_rdi = merged_df_years.groupby('Ctry_EditorialName')['RDI'].mean()

# Display the average RDI for each selected country
average_rdi.sample(10)

merged_df_years = pd.merge(merged_df_years, average_rdi, how='inner', left_on=['Ctry_EditorialName'], right_on=['Ctry_EditorialName'])

merged_df_years.sample(10)

composite_score =1

# Filter data for the latest available year
latest_year = merged_df_years['Year'].max()
test = merged_df_years[(merged_df_years['Year'] == latest_year) & 
                     ((merged_df_years['Composite Score'] > composite_score))]

# Create a choropleth map
fig = px.choropleth(merged_df_years, 
                    locations="Ctry_EditorialName", 
                    locationmode='ISO-3',
                    color="Composite Score", 
                    hover_name="Ctry_EditorialName",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="Countries with High Government Religious Restrictions")
# fig.show()
py.plot(fig, filename='world map', auto_open=True)

crime_indicators = ['Kidnappings per 100000 people',
    'Robberies per 100000 people',
    'Thefts per 100000 people',
    'Homicides per 100000 people'
    ]
# Step 2.1: Define threshold for high religious restrictions
median_restriction_score = normalized_scores2['Composite Score'].median()

# Classify countries into high and low religious restrictions groups
normalized_scores2['Religious Restriction Group'] = normalized_scores2['Composite Score'].apply(lambda x: 'High' if x >= 5 else 'Low')

# Combine the classification with average RDI and crime rates
classification = normalized_scores2[['Composite Score', 'Religious Restriction Group']]
classification['RDI'] = average_rdi
classification = classification.merge(merged_df_years.groupby('Ctry_EditorialName')[crime_indicators].mean(), left_index=True, right_index=True)

# Step 4: Compare crime rates between high restriction high diversity and high restriction low diversity groups
high_restriction = classification[classification['Religious Restriction Group'] == 'High']
high_restriction_high_diversity = high_restriction[high_restriction['RDI'] > high_restriction['RDI'].median()]
high_restriction_low_diversity = high_restriction[high_restriction['RDI'] <= high_restriction['RDI'].median()]

# Aggregate crime rates for high restriction high diversity and high restriction low diversity groups
crime_comparison_high_restriction = pd.DataFrame({
    'High Restriction High Diversity': high_restriction_high_diversity[crime_indicators].mean(),
    'High Restriction Low Diversity': high_restriction_low_diversity[crime_indicators].mean()
})
crime_comparison_high_restriction

# Combine the classification with average RDI and new crime rates
classification_low_restriction = normalized_scores2[['Composite Score', 'Religious Restriction Group']]
classification_low_restriction['RDI'] = average_rdi
classification_low_restriction = classification_low_restriction.merge(merged_df_years.groupby('Ctry_EditorialName')[crime_indicators].mean(), left_index=True, right_index=True)

# Filter for low religious restrictions
low_restriction = classification_low_restriction[classification_low_restriction['Religious Restriction Group'] == 'Low']
low_restriction_high_diversity = low_restriction[low_restriction['RDI'] > low_restriction['RDI'].median()]
low_restriction_low_diversity = low_restriction[low_restriction['RDI'] <= low_restriction['RDI'].median()]

# Aggregate crime rates for low restriction high diversity and low restriction low diversity groups
crime_comparison_low_restriction = pd.DataFrame({
    'Low Restriction High Diversity': low_restriction_high_diversity[crime_indicators].mean(),
    'Low Restriction Low Diversity': low_restriction_low_diversity[crime_indicators].mean()
})
crime_comparison_low_restriction

average_rdi

stats.ttest_ind(crime_comparison_high_restriction['High Restriction Low Diversity'], crime_comparison_low_restriction['Low Restriction High Diversity'] ,equal_var=False)

high_restriction_countries = merged_df_years[merged_df_years['Composite Score']> 5]
low_restriction_countries = merged_df_years[merged_df_years['Composite Score'] <=5]

merged_df_years

low_restriction_countries

high_restriction_countries

crime_rate_columns =['Kidnappings per 100000 people',
    'Robberies per 100000 people',
    'Thefts per 100000 people',
    'Homicides per 100000 people']
# Calculate the overall average crime rates for each country without using ace_tools
overall_average_crime_rates_by_country = merged_df_years.groupby('Ctry_EditorialName')[crime_rate_columns].mean()

# Display the result
overall_average_crime_rates_by_country.sample(10)

overall_average_crime_rates_by_country['Overall Average Crime Rates'] = overall_average_crime_rates_by_country.mean(axis=1)

overall_average_crime_rates_by_country

merged_df_years = pd.merge(merged_df_years, overall_average_crime_rates_by_country, how='inner', left_on=['Ctry_EditorialName'], right_on=['Ctry_EditorialName'])

merged_df_years

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
merged_scaled_df1 = scaler.fit_transform(merged_df_years[['GRI_y', 'SHI_y', 'RDI', 'People practicing Judaism as percent of the population',
    'Buddhists as percent of the total population',
    'People practicing Hinduism as percent of the population',
    'Muslims as percent of the total population',
    'Christians as percent of the total population',
    'Non religious people as percent of the population']])

analysis_colums = [['GRI_y', 'SHI_y', 'RDI', 'People practicing Judaism as percent of the population',
    'Buddhists as percent of the total population',
    'People practicing Hinduism as percent of the population',
    'Muslims as percent of the total population',
    'Christians as percent of the total population',
    'Non religious people as percent of the population']]

merged_scaled_df1 = pd.DataFrame(merged_scaled_df1, columns = analysis_colums)

merged_scaled_df1

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Select your dependent and independent variables
X = merged_scaled_df1[['GRI_y', 'SHI_y','RDI', 'People practicing Judaism as percent of the population',
    'Buddhists as percent of the total population',
    'People practicing Hinduism as percent of the population',
    'Muslims as percent of the total population',
    'Christians as percent of the total population',
    'Non religious people as percent of the population']]
y = merged_df_years['Overall Average Crime Rates']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
model = sm.OLS(y_train, X_train).fit()

# Print the model summary
print(model.summary())

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
r_squared = model.rsquared
print(f'R-squared: {r_squared}')