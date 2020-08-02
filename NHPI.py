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

NHPI = june[['State', 'NHPI_June15']]
NHPI = NHPI.dropna()

#Add July numberse
july_dict = dict(zip(july['State'], july['NHPI_July20']))
NHPI['NHPI_July20'] = NHPI['State']
NHPI = NHPI.replace({'NHPI_July20': july_dict})

#Add Reopening dates
rm_dict = dict(zip(rm['State'], rm['Reopening']))
NHPI['Reopening'] = NHPI['State']
NHPI = NHPI.replace({'Reopening': rm_dict})

#Add Mask dates
rm_dict = dict(zip(rm['State'], rm['Mask']))
NHPI['Mask'] = NHPI['State']
NHPI = NHPI.replace({'Mask': rm_dict})

#Eliminate states with no June data
NHPI = NHPI.loc[(NHPI['NHPI_June15'] > 0)]

#Eliminate states with no July data
NHPI = NHPI.loc[(NHPI['NHPI_July20'] > 0)]

#Add percent increase column
NHPI['Percent_change'] = ((NHPI['NHPI_July20'] - NHPI['NHPI_June15']) / NHPI['NHPI_June15']) * 100

#Eliminate states with negative percent change -- this can only happen if we are missing data
NHPI = NHPI.loc[(NHPI['Percent_change'] > 0)]

#Is the percent change >= to 100?
NHPI['over_100'] = (NHPI['Percent_change'] >= 100)

#Is the percent change >= to 200?
NHPI['over_200'] = (NHPI['Percent_change'] >= 200)

print(NHPI)

NHPI.to_csv('NHPI_data.csv')