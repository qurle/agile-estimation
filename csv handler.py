import pandas as pd
import matplotlib.pyplot as plt
import fitter as f
import numpy as np
import datetime


# presets
plt.style.use('seaborn')

# whole csv data
csv = u'data\grand dataset.csv'
df = pd.read_csv(csv, header=0, index_col=False, names = ['Organization', 'Project', 'Issue','Status', 'Assignee', 'Created', 'Resolved', 'SP', 'Time Spent'])
# sampling parameters
training_amount = .70
test_project = 'HUDI'

# Remove unnecessary data, create training and validation datasets
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
    training = data.sample(frac=training_amount, random_state=123)
    validation = data.drop(training.index)

    print('There are ' + str(training.shape[0]) + ' datapoints in training dataset')
    plt.title(label = (str(training.shape[0]) + ' datapoints of ' + project))

    return training

def plot_scatter(data):
    x = data['Time Spent']
    y = data['SP']
    plt.scatter(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r--')


# Write density info to txt file
def write_density(project, density):
    f = open('plots/densities.txt', 'a')
    f.write(project + ' (' + datetime.datetime.now().strftime('%d.%m %H:%M') + '):\n' + str(density))

# Call Fitter to find best fitting density distributions
def analyse_density(data, project, all_dist, plot):
    # checking density functions 
    fitter = f.Fitter(data['SP Cost'], distributions = f.get_distributions() if all_dist else f.get_common_distributions())
    fitter.fit()

    write_density(project, fitter.get_best(method = 'sumsquare_error') + '\n\n')
    if plot: fitter.summary()
    

def show_plot():
    plt.show()

def save_plot(project, folder = '', suffix=''):
    path = 'plots/'
    if folder: 
        path+= folder + '/'
    path+= project
    if suffix: 
        path+= folder + '/'
    path+='.png'
    plt.savefig('plots/'+project+'.png')

def count_issues():
    print(df.groupby('Project')['Issue'].nunique())

def iterate(all_dist=True):
    for project in df['Project'].unique():
        print(project)
        data = collect_data(project)
        analyse_density(data, project, all_dist, plot=True)
        save_plot(project)
    
def test(all_dist=False):
    print(test_project)
    data = collect_data(test_project)
    plot_scatter(data)
    show_plot()

for project in df['Project'].unique():
    print(test_project)
    data = collect_data(test_project)
    plot_scatter(data)
    save_plot(project)