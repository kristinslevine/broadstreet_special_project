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

hispanic = june[['State', 'Hispanic_June15']]
hispanic = hispanic.dropna()

#Add July numberse
july_dict = dict(zip(july['State'], july['Hispanic_July20']))
hispanic['Hispanic_July20'] = hispanic['State']
hispanic = hispanic.replace({'Hispanic_July20': july_dict})

#Add Reopening dates
rm_dict = dict(zip(rm['State'], rm['Reopening']))
hispanic['Reopening'] = hispanic['State']
hispanic = hispanic.replace({'Reopening': rm_dict})

#Add Mask dates
rm_dict = dict(zip(rm['State'], rm['Mask']))
hispanic['Mask'] = hispanic['State']
hispanic = hispanic.replace({'Mask': rm_dict})

#Eliminate states with no June data
hispanic = hispanic.loc[(hispanic['Hispanic_June15'] > 0)]

#Eliminate states with no July data
hispanic = hispanic.loc[(hispanic['Hispanic_July20'] > 0)]

#Add percent increase column
hispanic['Percent_change'] = ((hispanic['Hispanic_July20'] - hispanic['Hispanic_June15']) / hispanic['Hispanic_June15']) * 100

#Eliminate states with negative percent change -- this can only happen if we are missing data
hispanic = hispanic.loc[(hispanic['Percent_change'] > 0)]

#Is the percent change >= to 100?
hispanic['over_100'] = (hispanic['Percent_change'] >= 100)

print(hispanic)

hispanic.to_csv('hispanic_data.csv')