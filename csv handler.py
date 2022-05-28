from ctypes import resize
import pandas as pd
import matplotlib.pyplot as plt
import fitter as fit
import numpy as np
import scipy.stats as sc
import datetime
import math

# presets
plt.switch_backend('agg')
plt.style.use('seaborn')

# whole csv data
csv = u'data\grand dataset.csv'
df = pd.read_csv(csv, header=0, index_col=False, names = ['Organization', 'Project', 'Issue','Status', 'Assignee', 'Created', 'Resolved', 'SP', 'Time Spent'])
# output files
log_file = open('data/log ' + datetime.datetime.now().strftime('%m-%d %H%M') + '.txt', 'a+', encoding="utf-8")
# array for results (see calc_error())
result = []
# model parameters
training_amount = .70
low = .05
high = .95
repeats = 10000
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
# Logs project codename, all/common dist, found dist itself and time
def find_distribution(data, project, all_dist = False, plot = False, log = False):
    # checking density functions 
    fitter = fit.Fitter(data['SP Cost'], distributions = fit.get_distributions() if all_dist else fit.get_common_distributions())
    fitter.fit()
    dist = fitter.get_best(method = 'sumsquare_error')

    if plot: fitter.summary()
    if log: log_file.write(project + ',' + (' all' if all_dist else ' common') + ' (' + datetime.datetime.now().strftime('%d.%m %H:%M') + '):\n' + str(dist) + '\n\n')

    return dist

# Generate dataframe full of possible issue timings
# Logs datapoints count and dataframe summary 
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

    if log: log_file.write(str(mc.shape[1]-1) + ' datapoints:\n' + str(mc) + '\n\n')

    return mc

# Estimate in terms of low, median and high scores
def estimate(data : pd.DataFrame):
    optimistic = data['Sum'].quantile(low)
    median = data['Sum'].quantile(.5)
    pessimistic = data['Sum'].quantile(high)
    return median, pessimistic, optimistic

# Calculate error
# Logs info about real value and estimation
def calc_error(data : pd.DataFrame, m, p, o):
    real = data['Time Spent'].sum()
    e = m-real
    re = e/real*100
    in_range = o <= real <= p
    return e, re, in_range, real,

def assemble_results(project, dp, m, p, o, e, re, in_range, real, log=False):
    def r(int): return str(round(int))  # rounded to string
    dict = {
        'Project':      project, 
        'Datapoints':   r(dp), 
        'Time Spent':   r(real), 
        'Estimation':   r(m), 
        'Error':        r(e), 
        'Relative Error': r(re) + '%', 
        'Score':        '★' * min(math.floor((120-abs(re))/20),5) + '☆' * min(math.floor(abs(re)/20), 5),
        'Range':     '[' + r(o) + ' — ' + r(p) + ']',
        'In Range':  ('✔️' if in_range else '❌')
        }
    if log:    
        log_file.write(
                'Real: ' + dict['Time Spent'] + '\tEst: ' + dict['Estimation'] + 
                '\nError: ' + dict['Error'] + '\tRel Error:' + dict['Relative Error'] +  
                '\nRange: ' + dict['Range'] + ' ' + dict['In Range'] + ' ' + dict['Score'] + '\n\n'
                )
    return dict

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

# found dist for projects
dists = {
    'CAMEL':{'foldcauchy': {'c': 62.85052977637048, 'loc': 0.03198802705108442, 'scale': 19.097763407060896}},
    'HUDI':{'johnsonsb': {'a': 0.10375707614138747, 'b': 0.15024438216311276, 'loc': 299.31299682606635, 'scale': 14100.687003173936}},
    'MXNET':{'johnsonsu': {'a': -1.895228156812165, 'b': 0.811966263064406, 'loc': 481.21050446434776, 'scale': 432.3555997748377}},
    'CRUC':{'fisk': {'c': 1.5257516029035587, 'loc': 157.7927833674512, 'scale': 1034.9758856993885}},
    'FE':{'fatiguelife': {'c': 2.002449115781337, 'loc': -0.19842595736915364, 'scale': 1651.5776867647423}},
    'BE':{'kappa3': {'a': 1.9064296893959423, 'loc': 2399.999882864805, 'scale': 9411.278987434493}},
    'FAB':{'gennorm': {'beta': 0.2260531051688345, 'loc': 7200.0, 'scale': 5.801127164752151}},
    'DOCS':{'alpha': {'a': 3.710296399210331e-07, 'loc': 0.3551425301524438, 'scale': 0.2261247654069295}},
    'EVG':{'loglaplace': {'c': 1.6716767765334293, 'loc': 0.18009885760387134, 'scale': 0.31990114320418794}},
    'QT':{'arcsine': {'loc': -10306.147356979782, 'scale': 106306.14735697981}},
    'JBEAP':{'laplace_asymmetric': {'kappa': 0.25796143770215074, 'loc': 3600.0000000000055, 'scale': 5858.147281372981}},
    'RHODS':{'dgamma': {'a': 0.45628466050500815, 'loc': 3600.0000000000005, 'scale': 2787.601085285415}},
    'MULTIARCH':{'dgamma': {'a': 0.20920269730490693, 'loc': 28799.999999999996, 'scale': 77715.70960082884}},
    'ODC':{'geninvgauss': {'p': 0.44154098850759205, 'b': 3.996669259674054e-09, 'loc': 5759.999999999999, 'scale': 6.411108008811669e-05}},
    'TEIID':{'laplace_asymmetric': {'kappa': 0.6715210386208867, 'loc': 14399.999999999996, 'scale': 7400.353816269922}},
    'TEIIDSB':{'dgamma': {'a': 0.4462643738096519, 'loc': 14400.000000000002, 'scale': 16893.862900846198}}
}

def pick_distribution(project, dists, all_dist, log = False):
    dist = dists.get(project)
    if log: log_file.write(project + ',' + (' all' if all_dist else ' common') + ' (' + datetime.datetime.now().strftime('%d.%m %H:%M') + '):\n' + str(dist) + '\n\n')
    return dist


# Like, do it for all projects
def iterate(dists = False, all_dist=True, log = True):
    for project in df['Project'].unique():
        print(project)
        training, validation = collect_data(project)
        if dists:
            dist = pick_distribution(project, dists, all_dist = all_dist, log=log)
        else:
            dist = find_distribution(training, project, all_dist = all_dist, plot = True, log = log)
        mc = generate(validation, project, dist, repeats, log = log)
        estimation = estimate(mc)
        errors = calc_error(validation, *estimation)
        result.append(assemble_results(project, validation.shape[0], *estimation, *errors, log=log))
    log_file.close()
    result_df = pd.DataFrame(result)
    result_df.to_csv('data/results ' + datetime.datetime.now().strftime('%m-%d %H%M') + '.csv', index=False, encoding="utf_8_sig")
    
# Do it for test project
def test(all_dist=False):
    print(test_project)
    data = collect_data(test_project)
    plot_scatter(data)
    show_plot()

iterate(dists, all_dist=True, log=True)