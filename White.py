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

white = june[['State', 'White_June15']]
white = white.dropna()

#Add July numberse
july_dict = dict(zip(july['State'], july['White_July20']))
white['White_July20'] = white['State']
white = white.replace({'White_July20': july_dict})

#Add Reopening dates
rm_dict = dict(zip(rm['State'], rm['Reopening']))
white['Reopening'] = white['State']
white = white.replace({'Reopening': rm_dict})

#Add Mask dates
rm_dict = dict(zip(rm['State'], rm['Mask']))
white['Mask'] = white['State']
white = white.replace({'Mask': rm_dict})

#Eliminate states with no June data
white = white.loc[(white['White_June15'] > 0)]

#Eliminate states with no July data
white = white.loc[(white['White_July20'] > 0)]

#Add percent increase column
white['Percent_change'] = ((white['White_July20'] - white['White_June15'])/white['White_June15'])*100

#Eliminate states with negative percent change -- this can only happen if we are missing data
white = white.loc[(white['Percent_change'] > 0)]

#Is the percent change >= to 100?
white['over_100'] = (white['Percent_change'] >= 100)


print(white)

white.to_csv('white_data.csv')