from wsgiref import validate
import pandas as pd
import matplotlib.pyplot as plt
import fitter as f
import numpy as np
import scipy.stats as sc
import datetime


# presets
plt.style.use('seaborn')

# whole csv data
csv = u'data\grand dataset.csv'
df = pd.read_csv(csv, header=0, index_col=False, names = ['Organization', 'Project', 'Issue','Status', 'Assignee', 'Created', 'Resolved', 'SP', 'Time Spent'])
# model parameters
training_amount = .70
test_project = 'HUDI'
repeats = 10
seed = 123

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
    training = data.sample(frac=training_amount, random_state=seed)
    validation = data.drop(training.index)

    print('There are ' + str(training.shape[0]) + ' datapoints in training dataset')
    plt.title(label = (str(training.shape[0]) + ' datapoints of ' + project))

    return training, validation

# Write density info to txt file
def write_density(project, density):
    f = open('data/densities.txt', 'a')
    f.write(project + ' (' + datetime.datetime.now().strftime('%d.%m %H:%M') + '):\n' + str(density))

# Call Fitter to find best fitting density distributions
def analyse_density(data, project, all_dist, plot):
    # checking density functions 
    fitter = f.Fitter(data['SP Cost'], distributions = f.get_distributions() if all_dist else f.get_common_distributions())
    fitter.fit()
    density = fitter.get_best(method = 'sumsquare_error')
    write_density(project, density + '\n\n')
    if plot: fitter.summary()
    return density

# Draw scatters
def plot_scatter(data):
    x = data['Time Spent']
    y = data['SP']
    plt.xlabel('Time Spent')
    plt.ylabel('Estimated SP')
    plt.scatter(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r--')

def show_plot():
    plt.show()

# Save plots as plots/folder../project-suffix.png
def save_plot(project, folder = '', suffix=''):
    path = 'plots/'
    if folder: 
        path+= folder + '/'
    path+= project
    if suffix: 
        path+= suffix
    path+='.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def count_issues():
    print(df.groupby('Project')['Issue'].nunique())

def generating(data, project, dist, repeats = 10000):
    # Making sure our random stays predictable
    np.random.seed(seed=seed)
    # Creating special dataset to write up
    mcs = pd.DataFrame(columns=list(data.index.values))

    # Converting object to string 'sc.dist.rvs(key = value, key2 = ...)'
    dist_name = list(dist.keys())[0]
    dist_params = ''
    for k, v in list(dist.values())[0].items():
        dist_params+=k+'='+str(v)+', '
    # This string
    dist_string = 'sc.'+dist_name+'.rvs('+dist_params+'size='+str(repeats)+')'

    # Generating 
    for issue in mcs:
        sp_count = data.loc[issue, 'SP']
        print(eval(dist_string))
        mcs[issue] = eval(dist_string)*sp_count

# Write generated data to file
def save_mcs(project, data, mcs):
    f = open('data/generated data.txt', 'a')
    f.write(project + ', ' + str(data.shape[0]) + ' datapoints (' + 
    datetime.datetime.now().strftime('%d.%m %H:%M') + '):\n' + mcs.to_string() + '\n\n')

# Like, do it for all projects
def iterate(all_dist=True):
    for project in df['Project'].unique():
        print(project)
        training, validation = collect_data(project)
        analyse_density(training, project, all_dist, plot=True)
        save_plot(project)
    
# Do it for test project
def test(all_dist=False):
    print(test_project)
    data = collect_data(test_project)
    plot_scatter(data)
    show_plot()

training, validation = collect_data(test_project)
generating(validation, test_project, repeats=repeats)
