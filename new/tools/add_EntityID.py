import os
import pandas as pd
import re
df = pd.read_csv('data/all_file.csv',index_col='entity')
for i, index in enumerate(df.index.unique()):
    df.loc[index, 'entityID'] = int(i)
df.reset_index(inplace=True)
df['entityID']=df['entityID'].astype("int64")
print(df)
df.to_csv("data/all_file1.csv")