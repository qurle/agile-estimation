import pandas as pd
import matplotlib.pyplot as plt
import fitter as f

# whole csv data
csv = u'data\grand dataset.csv'
df = pd.read_csv(csv, header=0, index_col=False, names = ['Organization', 'Project', 'Issue','Status', 'Assignee', 'Created', 'Resolved', 'SP', 'Time Spent'])
# sampling parameters
trainingAmount = .70
project = 'MXNET'

# for project in df['Project'].unique():

def collect_data(project):
    # actual data
    data = pd.DataFrame(df[df['Project'] == project], columns= ['Issue', 'SP', 'Time Spent'])
    data.set_index('Issue', inplace=True)
    data['SP Cost'] = data['Time Spent'] / data['SP'] 
    # splitting to training and validation dataset
    training = data.sample(frac=trainingAmount)
    validation = data.drop(training.index)
    print('There are ' + str(training.shape[0]) + ' datapoints in training dataset')

    fitter = f.Fitter(training['SP Cost'], distributions = f.get_common_distributions())
    fitter.fit()
    print(fitter.get_best(method = 'sumsquare_error'))
    fitter.summary()

def show_plot(data, project):
    plt.title(label = (str(data.shape[0]) + ' datapoints of ' + project))
    plt.show()

def save_plot(project):
    plt.savefig('plots/'+project+'.png')

def count_issues():
    print(df.groupby('Project')['Issue'].nunique())
