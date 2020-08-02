import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',14)

july = pd.read_csv('July_20.csv')
print(july.head(10))

print("The columns in this dataframe are:")
print(list(july))
print('\n')
print("The shape of the dataframe is:", july.shape)
print('\n')
print(july.dtypes)
print('\n')

july['County'] = july['County'].astype('str')
new = july['County'].str.split(',', n = 1, expand = True)
july['County'] = new[0]
july['State'] = new[1]


july['White'] = july['White'].str.replace(",", "")
july['Black'] = july['Black'].str.replace(",", "")
july['Black'] = july['Black'].str.replace("<", "")
july['Asian'] = july['Asian'].str.replace(",", "")
july['Asian'] = july['Asian'].str.replace("<", "")
july['AIAN'] = july['AIAN'].str.replace(",", "")
july['AIAN'] = july['AIAN'].str.replace("<", "")
july['NHPI'] = july['NHPI'].str.replace(",", "")
july['Hispanic'] = july['Hispanic'].str.replace(",", "")
july['Hispanic'] = july['Hispanic'].str.replace(" ", "")

july = july.fillna(0)
july = july.replace(to_replace='-', value ='0')

july['State'] = july['State'].str.replace(".", "")
july['State'] = july['State'].str.replace(" ", "")

print(july.head(10))
print(july.tail(10))
print("\n")

#Sum per state of white cases on July 20

july['White'] = july['White'].astype(str).astype(float)
july['State'] = july['State'].astype(str)

july_white = july.groupby('State').agg({'White': 'sum'}).reset_index()
july_white = july_white.rename(columns = {'White': 'White_July20'})
print(july_white.round())
print("\n")

#Sum per state of black cases on July 20

july['Black'] = july['Black'].astype(str).astype(float)
july['State'] = july['State'].astype(str)

july_black = july.groupby('State').agg({'Black': 'sum'}).reset_index()
july_black = july_black.rename(columns = {'Black': 'Black_July20'})
print(july_black.round())
print("\n")

#Sum per state of asian cases on July 20

july['Asian'] = july['Asian'].astype(str).astype(float)
july['State'] = july['State'].astype(str)

july_asian = july.groupby('State').agg({'Asian': 'sum'}).reset_index()
july_asian = july_asian.rename(columns = {'Asian': 'Asian_July20'})
print(july_asian.round())
print("\n")

#Sum per state of AIAN cases on July 20

july['AIAN'] = july['AIAN'].astype(str).astype(float)
july['State'] = july['State'].astype(str)

july_AIAN = july.groupby('State').agg({'AIAN': 'sum'}).reset_index()
july_AIAN = july_AIAN.rename(columns = {'AIAN': 'AIAN_July20'})
print(july_AIAN.round())
print("\n")

#Sum per state of NHPI cases on July 20

july['NHPI'] = july['NHPI'].astype(str).astype(float)
july['State'] = july['State'].astype(str)

july_NHPI = july.groupby('State').agg({'NHPI': 'sum'}).reset_index()
july_NHPI = july_NHPI.rename(columns = {'NHPI': 'NHPI_July20'})
print(july_NHPI.round())
print("\n")

#Sum per state of hispanic cases on July 20

july['Hispanic'] = july['Hispanic'].astype(str).astype(float)
july['State'] = july['State'].astype(str)

july_hispanic = july.groupby('State').agg({'Hispanic': 'sum'}).reset_index()
july_hispanic = july_hispanic.rename(columns = {'Hispanic': 'Hispanic_July20'})
print(july_hispanic.round())
print("\n")

#Putting all into one dataframe

july_total = july_white
july_total['Black_July20'] = july_black['Black_July20']
july_total['Asian_July20'] = july_asian['Asian_July20']
july_total['AIAN_July20'] = july_AIAN['AIAN_July20']
july_total['NHPI_July20'] = july_NHPI['NHPI_July20']
july_total['Hispanic_July20'] = july_hispanic['Hispanic_July20']
print(july_total)

july_total.to_csv('july_total.csv')