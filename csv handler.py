from statistics import median
from wsgiref import validate
import pandas as pd
import matplotlib.pyplot as plt
import fitter as f
import numpy as np
import scipy.stats as sc
import datetime
import math

# presets
plt.style.use('seaborn')

# whole csv data
csv = u'data\grand dataset.csv'
df = pd.read_csv(csv, header=0, index_col=False, names = ['Organization', 'Project', 'Issue','Status', 'Assignee', 'Created', 'Resolved', 'SP', 'Time Spent'])
# model parameters
training_amount = .70
low = .05
high = .95
repeats = 10
seed = 123
test_project = 'HUDI'

def count_issues():
    print(df.groupby('Project')['Issue'].nunique())

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

# Call Fitter to find best fitting density distributions
def find_distribution(data, project, all_dist, plot):
    # checking density functions 
    fitter = f.Fitter(data['SP Cost'], distributions = f.get_distributions() if all_dist else f.get_common_distributions())
    fitter.fit()
    density = fitter.get_best(method = 'sumsquare_error')
    if plot: fitter.summary()
    return density

# Write density info to txt file
def write_density(project, density):
    f = open('data/densities.txt', 'a')
    f.write(project + ' (' + datetime.datetime.now().strftime('%d.%m %H:%M') + '):\n' + str(density))

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

def generate(data, project, dist, repeats = 10000, log = False):
    # Making sure our random stays predictable
    np.random.seed(seed=seed)
    # Creating special dataset to write up
    mc = pd.DataFrame(columns=list(data.index.values))

    # Converting object to string 'sc.dist.rvs(key = value, key2 = ...)'
    dist_name = list(dist.keys())[0]
    dist_params = ''
    for k, v in list(dist.values())[0].items():
        dist_params+=k+'='+str(v)+', '
    # This string
    dist_string = 'sc.'+dist_name+'.rvs('+dist_params+'size='+str(repeats)+')'

    # Generating 
    for issue in mc:
        sp_count = data.loc[issue, 'SP']
        mc[issue] = eval(dist_string)*sp_count
    
    # Caluculating the sum
    mc['Sum'] = mc.sum(axis=1)

    if log:
        f = open('data/generated data.txt', 'a')
        f.write(project + ', ' + str(mc.shape[1]-1) + ' datapoints (' + datetime.datetime.now().strftime('%d.%m %H:%M') + '):\n' + str(mc) + '\n\n')
        f.close()

    return mc

# Estimate in terms of low, median and high scores
def estimate(data : pd.DataFrame):
    optimistic = data['Sum'].quantile(low)
    median = data['Sum'].quantile(.5)
    pessimistic = data['Sum'].quantile(high)
    return median, pessimistic, optimistic

# Calculate error
def calc_error(data : pd.DataFrame, m, p, o, log = False):
    real = data['Time Spent'].sum()
    e = m-real
    re = e/real*100
    in_range = o <= real <= p

    if log:    
        def r(int): return str(round(int))  # rounded to string
        f = open('data/generated data.txt', 'a')
        f.write('Real: ' + r(real) + '\tEst: ' + r(m) + 
                '\nError: ' + r(e) + '\tRel Error:' + r(re) + '%\t' '★'*math.floor(100/re) + '☆'+ math.floor(re/20)+
                '\nRange: [' + r(o) + ' — ' + r(p) + ']\t' + '✔️' if in_range else '❌')

    return e, in_range

# Like, do it for all projects
def iterate(all_dist=True):
    for project in df['Project'].unique():
        print(project)
        training, validation = collect_data(project)
        dens = find_distribution(training, project, all_dist, plot=True)
        write_density(project, dens)
        save_plot(project)
    
# Do it for test project
def test(all_dist=False):
    print(test_project)
    data = collect_data(test_project)
    plot_scatter(data)
    show_plot()

training, validation = collect_data(test_project)
dist = find_distribution(training, test_project, False, True)
mc = generate(validation, test_project, dist, repeats=repeats)
calc_error(validation, *estimate(mc))