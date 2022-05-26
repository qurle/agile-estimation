import pandas as pd
import matplotlib.pyplot as plt
import fitter as f
import datetime


# presets
plt.style.use('seaborn')

# whole csv data
csv = u'data\grand dataset.csv'
df = pd.read_csv(csv, header=0, index_col=False, names = ['Organization', 'Project', 'Issue','Status', 'Assignee', 'Created', 'Resolved', 'SP', 'Time Spent'])
# sampling parameters
training_amount = .70
test_project = 'HUDI'

def write_density(project, density):
    f = open('plots/densities.txt', 'a')
    f.write(project + ' (' + datetime.datetime.now().strftime('%d.%m %H:%M') + '):\n' + str(density))

def collect_data(project):
    # actual data
    data = pd.DataFrame(df[df['Project'] == project], columns= ['Issue', 'SP', 'Time Spent'])
    data.set_index('Issue', inplace=True)

    # when we have 0 SP issues and don't wanna infinity
    if (data['SP'] ==0).any():
        data['SP']= data['SP'] + 1

    # adding calculated column of 1 SP cost
    data['SP Cost'] = data['Time Spent'] / data['SP'] 

    # splitting to training and validation dataset
    training = data.sample(frac=training_amount)
    validation = data.drop(training.index)
    print('There are ' + str(training.shape[0]) + ' datapoints in training dataset')

    # checking density functions 
    fitter = f.Fitter(training['SP Cost'], distributions = f.get_distributions())
    fitter.fit()
    write_density(project, fitter.get_best(method = 'sumsquare_error'))
    fitter.summary()
    plt.title(label = (str(data.shape[0]) + ' datapoints of ' + project))
    return training

def show_plot():
    plt.show()

def save_plot(project):
    plt.savefig('plots/'+project+'.png')

def count_issues():
    print(df.groupby('Project')['Issue'].nunique())

for project in df['Project'].unique():
    print(project)
    collect_data(project)
    save_plot(project)