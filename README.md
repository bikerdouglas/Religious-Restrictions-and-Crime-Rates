# Religious-Restrictions-and-Crime-Rates# Religion and Crime Rates Analysis

## Table of Contents

- Introduction
- Setup
- Data
- Exploration and Analysis
- Conclusion


# Introduction

This project aims to explore the relationship between the religious restrictions of various countries from around the world and their crime rates. The analysis is performed using data on different religions and crime rates while also factoring in the religious restrictions and diversity of each country. The primary objective of this project is to understand how religious restrictions, religion, and religous divery of a country factor into crime rates. 


# Setup

To run the analysis, ensure you have the following libraries installed:

 pandas
 numpy 
 matplotlib.pyplot 
 seaborn
 plotly.express

# Data

The [dataset](</Users/michaelphillipacosta/Desktop/18-06-24 09_42_58_theglobaleconomy.csv>) provided by The Global Economy includes: Kidnappings per 100,000 people, Robberies per 100,000 people, Homicides per 100,000 people, Burglaries per 100,000 people, and Prisoners per 100,000 people. These data points are segmented by year and country, alongside the religious composition of the population. The raw dataset had a range from 1960 - present. The [dataset](</Users/michaelphillipacosta/Desktop/PublicDataSet_ReligiousRestrictions_2007to2016.dta>) provided by Pew Research includes data on both government restricions and social hindrance indicators, also segmented by year and country. The range for this data set is from 2007-2016.
I was able to merge the datasets on the name the countries name and the year. I used a proportional imputation function using the mean to fill the missing value of the religious colummns first. I then filtered the data for 2009-2013 and that took care of any nan values I had.  


# Exploration and Analysis

## Average Crimes Rates and Religious Composition
After importing and cleaning my data using 'pandas', I wanted to get a futher understanding of the data by getting the religious restrictioins and religous compositions for each country. the attribute i chose to use were the religous columns, crime rate columns, the Government Restriction Index 'GRI' and Scoail Hindrance Index 'SHI'to get a religious restriction score, and the religous diversity of the country following the steps provided by pew research.

## Defining thresholds and Comparing 
The next goal was to calculate a religious restriction score. I used the 'GRI' and 'SHI' to get a composite score for each country and used that to plot an [interactive map](<https://plotly.com/~bikerdouglas/45/>). 


define thresholds for categorizing countries based on the percentage of the population adhering to different religions. threshold were defined as: 0-20 as low, 20-60 as mediums, and 60-100 as high. Once the religious compositions were defined. I wanted to get the average crime rates by religious composition and compare. The following bar graphs provide some insight into how the religious composition of countries compare to each other based on the crime indicators.

![Online Image](https://github.com/bikerdouglas/religions_crimes/blob/main/images/graphs/muslim_2.png?raw=true)
![Online Image](https://github.com/bikerdouglas/religions_crimes/blob/main/images/graphs/christianity_2.png?raw=true)
![Online Image](https://github.com/bikerdouglas/religions_crimes/blob/main/images/graphs/non-religious_2.png?raw=true)
![Online Image](https://github.com/bikerdouglas/religions_crimes/blob/main/images/graphs/hinduism_2.png?raw=true)
![Online Image](https://github.com/bikerdouglas/religions_crimes/blob/main/images/graphs/buddhism_2.png?raw=true)
![Online Image](https://github.com/bikerdouglas/religions_crimes/blob/main/images/graphs/judaism_2.png?raw=true)

## Comparing Religious Compositions Against Each Other
Finally I wanted to compare the different religious compositions against each other. the following graph shows the relationship between the religious composition of a country's population and its crime rates. Each religious group is analyzed separately, and the crime rates are compared across different categories.
![Online Image](https://github.com/bikerdouglas/religions_crimes/blob/main/images/graphs/all_religions_2.png?raw=true)

# Conclusion
Lastly, was to check for a correlation between the religious composition and crime rates. The result is a bar chart that visualizes the correlation between the religious composition of countries and their overall average crime rate. Each bar represents a different religion, and the height of the bar indicates the correlation value. This visualization helps to easily compare how the percentage of each religion in the population correlates with crime rates. A positive value indicates a positive correlation. As the percentage of the religion increases, the overall average crime rate tends to increase. A negative value indicates a negative correlation. As the percentage of the religion increases, the overall average crime rate tends to decrease. A value close to zero indicates little to no correlation.
![Online Image](https://github.com/bikerdouglas/religions_crimes/blob/main/images/graphs/correlation_2.png?raw=true)