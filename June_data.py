import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',14)

june = pd.read_csv('June_15.csv')
print(june.head(10))

print("The columns in this dataframe are:")
print(list(june))
print('\n')
print("The shape of the dataframe is:", june.shape)
print('\n')
print(june.dtypes)
print('\n')

june['County'] = june['County'].astype('str')
new = june['County'].str.split(',', n = 1, expand = True)
june['County'] = new[0]
june['State'] = new[1]

june = june.fillna(0)
june = june.replace(to_replace='-', value ='0')


june['White'] = june['White'].str.replace(",", "")
june['Black'] = june['Black'].str.replace(",", "")
june['Black'] = june['Black'].str.replace("<", "")
june['Asian'] = june['Asian'].str.replace(",", "")
june['Asian'] = june['Asian'].str.replace("<", "")
june['AIAN'] = june['AIAN'].str.replace(",", "")
june['AIAN'] = june['AIAN'].str.replace("<", "")
june['NHPI'] = june['NHPI'].str.replace(",", "")
june['Hispanic'] = june['Hispanic'].str.replace(",", "")

june['State'] = june['State'].str.replace(".", "")
june['State'] = june['State'].str.replace(" ", "")

print(june.head(10))
print(june.tail(10))
print("\n")

#Sum per state of white cases on June 15

june['White'] = june['White'].astype(str).astype(float)
june['State'] = june['State'].astype(str)

june_white = june.groupby('State').agg({'White': 'sum'}).reset_index()
june_white = june_white.rename(columns = {'White': 'White_June15'})
print(june_white.round())
print("\n")

#Sum per state of black cases on June 15

june['Black'] = june['Black'].astype(str).astype(float)
june['State'] = june['State'].astype(str)

june_black = june.groupby('State').agg({'Black': 'sum'}).reset_index()
june_black = june_black.rename(columns = {'Black': 'Black_June15'})
print(june_black.round())
print("\n")

#Sum per state of asian cases on June 15

june['Asian'] = june['Asian'].astype(str).astype(float)
june['State'] = june['State'].astype(str)

june_asian = june.groupby('State').agg({'Asian': 'sum'}).reset_index()
june_asian = june_asian.rename(columns = {'Asian': 'Asian_June15'})
print(june_asian.round())
print("\n")

#Sum per state of AIAN cases on June 15

june['AIAN'] = june['AIAN'].astype(str).astype(float)
june['State'] = june['State'].astype(str)

june_AIAN = june.groupby('State').agg({'AIAN': 'sum'}).reset_index()
june_AIAN = june_AIAN.rename(columns = {'AIAN': 'AIAN_June15'})
print(june_AIAN.round())
print("\n")

#Sum per state of NHPI cases on June 15

june['NHPI'] = june['NHPI'].astype(str).astype(float)
june['State'] = june['State'].astype(str)

june_NHPI = june.groupby('State').agg({'NHPI': 'sum'}).reset_index()
june_NHPI = june_NHPI.rename(columns = {'NHPI': 'NHPI_June15'})
print(june_NHPI.round())
print("\n")

#Sum per state of hispanic cases on June 15

june['Hispanic'] = june['Hispanic'].astype(str).astype(float)
june['State'] = june['State'].astype(str)

june_hispanic = june.groupby('State').agg({'Hispanic': 'sum'}).reset_index()
june_hispanic = june_hispanic.rename(columns = {'Hispanic': 'Hispanic_June15'})
print(june_hispanic.round())
print("\n")

#Putting all into one dataframe

june_total = june_white
june_total['Black_June15'] = june_black['Black_June15']
june_total['Asian_June15'] = june_asian['Asian_June15']
june_total['AIAN_June15'] = june_AIAN['AIAN_June15']
june_total['NHPI_June15'] = june_NHPI['NHPI_June15']
june_total['Hispanic_June15'] = june_hispanic['Hispanic_June15']
print(june_total)

june_total.to_csv('june_total.csv')