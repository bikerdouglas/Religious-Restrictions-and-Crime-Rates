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
After importing and cleaning my data using 'pandas', I wanted to get a futher understanding of the data by getting the religious restrictioins and religous compositions for each country. the attribute i chose to use were the religous columns, crime rate columns, the Government Restriction Index 'GRI' and Scoail Hindrance Index 'SHI'to get a religious restriction score, and the religous diversity 'RDI' of the country following the steps provided by pew research.

## Calculating Religious Restrictions, Religious Diversity, and Overall Average Crime Rates
The next goal was to calculate a religious restriction score. I used the 'GRI' and 'SHI' to get a composite score for each country and used that to plot an interactive map. ![interactive map](https://github.com/bikerdouglas/Religious-Restrictions-and-Crime-Rates/blob/main/images/map.png?raw=true)
Next I was able to get the religious diversity scores using the steps provided by pew research. Finally I used the crime columns to get an overall average crime rates.  I used all three of these calculation to compare countries with high religious restrictions and low religious diversity against countries that have low religious restrictions and high religous diversity. High being any country with a score equal or greater than five, low being less than five. What i found was that countries with high restrictions and low diversity actually had lower overall crime rates than countries with low restrictions and high diversity. 

## Correlations
Before my using a machine model on the data I wanted to see if there were any correlations with the overall average crime rates and the 'GRI', 'SHI', 'RDI', and religious attributes. What I found was that from the religous attribute Non-religious practicers and Christains had the high positive correlation and the highest negative correlations belonging to Muslims and Buddhists respectively. Both 'GRI' and 'SHI' had negative correlations and 'RDI' had a slight positive correlation. 
![Online Image](https://github.com/bikerdouglas/Religious-Restrictions-and-Crime-Rates/blob/main/images/heatmap.png?raw=true)

## Linear Regression 
Using a linear regression model predicting the overall average crimes using the 'GRI', 'SHI', 'RDI', and relgious attributes I was able to find that Non-religious practicers, Christians, and 'SHI' were associated with a predicition of increasing overall average crime rates. Non-religous practicers had the highest coefficient, followed by Christians, and lastly 'SHI'. All other religous attributes were associated with decrease in overall average crime rate, with Muslims and Buddhists with highest negative coefficients respectively. the 'GRI' and 'RDI' also had negative coefficients. ![Online Image](https://github.com/bikerdouglas/Religious-Restrictions-and-Crime-Rates/blob/main/images/linear_reg_output.png?raw=true)

# Conclusion
The analysis of the relationship between religious restrictions, religious diversity, and crime rates across various countries provides insightful results. By merging datasets from The Global Economy and Pew Research Center, and calculating the Government Restriction Index (GRI), Social Hindrance Index (SHI), and Religious Diversity Index (RDI), I derived a comprehensive view of how these factors interplay with crime rates. Thees findings indicate that countries with high religious restrictions and low religious diversity tend to have lower overall crime rates compared to countries with low religious restrictions and high religious diversity. This counterintuitive result suggests that strict religious environments might be associated with more social order, potentially due to the influence of religious norms and community cohesion. The correlation analysis further revealed that non-religious practitioners and Christians have a positive correlation with higher crime rates, while Muslims and Buddhists have a negative correlation. Both GRI and SHI were negatively correlated with crime rates, implying that higher governmental and social restrictions on religion might contribute to lower crime rates. Conversely, RDI showed a slight positive correlation, indicating that more religiously diverse societies might experience slightly higher crime rates. In the linear regression model, non-religious practitioners, Christians, and SHI were associated with an increase in crime rates. This suggests that in environments with less religious affiliation or more social hindrances, crime rates may be higher. In contrast, Muslims, Buddhists, GRI, and RDI were associated with a decrease in crime rates, reinforcing the idea that religious adherence and governmental restrictions might contribute to social stability. Further more, These findings highlight the complex and multifaceted relationship between religion and crime. While strict religious restrictions and lower diversity seem to correlate with lower crime rates, the underlying social dynamics and cultural contexts play a crucial role in shaping these outcomes. Further research is needed to unpack these relationships and understand the mechanisms driving these associations.
