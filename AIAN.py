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

AIAN = june[['State', 'AIAN_June15']]
AIAN = AIAN.dropna()

#Add July numberse
july_dict = dict(zip(july['State'], july['AIAN_July20']))
AIAN['AIAN_July20'] = AIAN['State']
AIAN = AIAN.replace({'AIAN_July20': july_dict})

#Add Reopening dates
rm_dict = dict(zip(rm['State'], rm['Reopening']))
AIAN['Reopening'] = AIAN['State']
AIAN = AIAN.replace({'Reopening': rm_dict})

#Add Mask dates
rm_dict = dict(zip(rm['State'], rm['Mask']))
AIAN['Mask'] = AIAN['State']
AIAN = AIAN.replace({'Mask': rm_dict})

#Eliminate states with no June data
AIAN = AIAN.loc[(AIAN['AIAN_June15'] > 0)]

#Eliminate states with no July data
AIAN = AIAN.loc[(AIAN['AIAN_July20'] > 0)]

#Add percent increase column
AIAN['Percent_change'] = ((AIAN['AIAN_July20'] - AIAN['AIAN_June15']) / AIAN['AIAN_June15']) * 100

#Eliminate states with negative percent change -- this can only happen if we are missing data
AIAN = AIAN.loc[(AIAN['Percent_change'] > 0)]

#Is the percent change >= to 100?
AIAN['over_100'] = (AIAN['Percent_change'] >= 100)


print(AIAN)

AIAN.to_csv('AIAN_data.csv')