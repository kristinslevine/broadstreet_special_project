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

black = june[['State', 'Black_June15']]
black = black.dropna()

#Add July numberse
july_dict = dict(zip(july['State'], july['Black_July20']))
black['Black_July20'] = black['State']
black = black.replace({'Black_July20': july_dict})

#Add Reopening dates
rm_dict = dict(zip(rm['State'], rm['Reopening']))
black['Reopening'] = black['State']
black = black.replace({'Reopening': rm_dict})

#Add Mask dates
rm_dict = dict(zip(rm['State'], rm['Mask']))
black['Mask'] = black['State']
black = black.replace({'Mask': rm_dict})

#Eliminate states with no June data
black = black.loc[(black['Black_June15'] > 0)]

#Eliminate states with no July data
black = black.loc[(black['Black_July20'] > 0)]

#Add percent increase column
black['Percent_change'] = ((black['Black_July20'] - black['Black_June15']) / black['Black_June15']) * 100

#Eliminate states with negative percent change -- this can only happen if we are missing data
black = black.loc[(black['Percent_change'] > 0)]

#Is the percent change >= to 50?
black['over_50'] = (black['Percent_change'] >= 50)

#Is the percent change >= to 100?
black['over_100'] = (black['Percent_change'] >= 100)

print(black)

black.to_csv('black_data.csv')