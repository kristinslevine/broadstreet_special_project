import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',14)

june = pd.read_csv('June_total.csv')
july = pd.read_csv('July_total.csv')

rm = pd.read_csv('Reopen_Mask.csv')
rm = rm[['State', 'Reopening', 'Mask']]
rm = rm.fillna(0)
rm['State'] = rm['State'].str.replace(" ", "")

print('\n')
#print(june)
print('\n')
#print(july)
print('\n')
#print(rm)
print('\n')

asian = june[['State', 'Asian_June15']]
asian = asian.dropna()

#Add July numberse
july_dict = dict(zip(july['State'], july['Asian_July20']))
asian['Asian_July20'] = asian['State']
asian = asian.replace({'Asian_July20': july_dict})

#Add Reopening dates
rm_dict = dict(zip(rm['State'], rm['Reopening']))
asian['Reopening'] = asian['State']
asian = asian.replace({'Reopening': rm_dict})

#Add Mask dates
rm_dict = dict(zip(rm['State'], rm['Mask']))
asian['Mask'] = asian['State']
asian = asian.replace({'Mask': rm_dict})

#Eliminate states with no June data
asian = asian.loc[(asian['Asian_June15'] > 0)]

#Eliminate states with no July data
asian = asian.loc[(asian['Asian_July20'] > 0)]

#Add percent increase column
asian['Percent_change'] = ((asian['Asian_July20'] - asian['Asian_June15']) / asian['Asian_June15']) * 100

#Eliminate states with negative percent change -- this can only happen if we are missing data
asian = asian.loc[(asian['Percent_change'] > 0)]

#Is the percent change >= to 100?
asian['over_100'] = (asian['Percent_change'] >= 100)


print(asian)

asian.to_csv('asian_data.csv')