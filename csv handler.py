import csv
import pandas as pd 

# define path to data
PATH = u'data\grand dataset.csv'

# create panda datafrmae
data = pd.read_csv(PATH, header=0, names = ['Organization', 'Project', 'Issue','Status', 'Assignee', 'Created', 'Resolved', 'SP', 'Time Spent'])
data = data.groupby('Project')['Issue'].nunique()
print(data)
