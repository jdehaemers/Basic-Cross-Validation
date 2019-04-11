import numpy as np
import pandas as pd


data = pd.read_csv(r'CSVs/StudentsPerformance.csv', header=0)

data.rename(columns = {'race/ethnicity' : 'ethnicity'}, inplace=True)
data.rename(columns = {'parental level of education' : 'parent_ed'}, inplace=True)
data.rename(columns = {'test preparation course' : 'test_prep'}, inplace=True)
data.rename(columns = {'math score' : 'math_score'}, inplace=True)
data.rename(columns = {'reading score' : 'reading_score'}, inplace=True)
data.rename(columns = {'writing score' : 'writing_score'}, inplace=True)


obs = len(data)
data['prep_course'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'test_prep'] == 'completed':
        data.at[i, 'prep_course'] = 1

data['free_lunch'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'lunch'] == 'free/reduced':
        data.at[i, 'free_lunch'] = 1

data['male'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'gender'] == 'male':
        data.at[i, 'male'] = 1

data['eth_B'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'ethnicity'] == 'group D':
        data.at[i, 'eth_B'] = 1

data['eth_C'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'ethnicity'] == 'group C':
        data.at[i, 'eth_C'] = 1

data['eth_D'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'ethnicity'] == 'group D':
        data.at[i, 'eth_D'] = 1

data['eth_E'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'ethnicity'] == 'group E':
        data.at[i, 'eth_E'] = 1

data['ped_shs'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'parent_ed'] == 'some high school':
        data.at[i, 'ped_shs'] = 1

data['ped_hs'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'parent_ed'] == 'high school':
        data.at[i, 'ped_hs'] = 1

data['ped_sc'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'parent_ed'] == 'some college':
        data.at[i, 'ped_sc'] = 1

data['ped_ad'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'parent_ed'] == 'associate\'s degree':
        data.at[i, 'ped_ad'] = 1

data['ped_md'] = np.zeros((obs))
for i in range(obs):
    if data.at[i, 'parent_ed'] == 'master\'s degree':
        data.at[i, 'ped_md'] = 1


print(data.shape)
print(data.dtypes)
print(data.head(10))

data.to_csv(r'CSVs/exam_performance.csv')
