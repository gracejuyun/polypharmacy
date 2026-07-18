import pandas as pd
import os

os.chdir('')# working directory

df1=pd.read_csv('aki_initial.csv', header=0, sep=',')
df1_filt=df1[pd.to_datetime(df1['search_start'])<='2023-12-31']
df1_filt.to_csv('../aki_filtered.csv', index=False)

df2=pd.read_csv('mace_initial.csv', header=0, sep=',')
df2_filt=df2[pd.to_datetime(df2['search_start'])<='2023-12-31']
df2_filt.to_csv('../mace_filtered.csv', index=False)

df3=pd.read_csv('itching_inital.csv', header=0, sep=',')
df3_filt=df3[pd.to_datetime(df3['search_start'])<='2023-12-31']
df3_filt.to_csv('../itching_filtered.csv', index=False)

df4=pd.read_csv('urticaria_initial.csv', header=0, sep=',')
df4_filt=df4[pd.to_datetime(df4['search_start'])<='2023-12-31']
df4_filt.to_csv('../urticaria_filtered.csv', index=False)

