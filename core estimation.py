import pandas as pd
import matplotlib.pyplot as plt
import fitter as fit
import numpy as np
import scipy.stats as sc
import datetime
from tkinter import filedialog as fd


# presets
plt.switch_backend('agg')
plt.style.use('seaborn')


# output files
log_file = open('core/log ' + datetime.datetime.now().strftime('%m-%d %H%M') + '.txt', 'a+', encoding="utf-8")
plot_file= 'core/plot ' + datetime.datetime.now().strftime('%m-%d %H%M') + '.png'
# model parameters
columns = ['Issue', 'SP', 'Time Spent']
low = .05
high = .95
repeats = 10000
seed = 123
log = True
plot = True
all_dist = True
found_dist = False
# resource in man-hours per week (for example)
resource = 200

# Open files, remove unnecessary data, count SP cost
def collect_data():
    # open file with training data:
    print('Select training data. It should be CSV file with [' + columns[0]+ '], [' + columns[1]  + '] and [' + columns[2]+ '] columns')
    hist_file = fd.askopenfilename(title = "Select training data file",filetypes = (("CSV Files","*.csv"),),initialdir ='.')
    # create dataframe
    hist = pd.read_csv(hist_file, header=0, usecols = columns, index_col=columns[0])
    
    # logging and printing stuff
    if log: 
        log_file.write('Hist data file: ' + hist_file +'\n')
        log_file.write(str(hist) +'\n\n')
    print(hist)
    print('There are ' + str(hist.shape[0]) + ' datapoints in training dataset')
    plt.title(label = (str(hist.shape[0]) + ' datapoints'))

    # when we have 0 SP issues and don't wanna infinity
    if (hist['SP'] == 0).any():
        hist['SP'] = hist['SP'] + 1

    # adding calculated column of 1 SP cost
    hist['SP Cost'] = hist['Time Spent'] / hist['SP'] 
    
    # open file with new data:
    print('Select new data. It should be CSV file with [' + columns[0]+ '], [' + columns[1]  + '] columns')
    new_file = fd.askopenfilename(title = "Select new data file",filetypes = (("CSV Files","*.csv"),),initialdir ='.')
    # create dataframe
    new = pd.read_csv(new_file, header=0, usecols = columns[:1], index_col=columns[0])

    return hist, new

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
# Logs all/common dist, found dist itself and time
def find_distribution(data):
    # checking density functions 
    fitter = fit.Fitter(data['SP Cost'], distributions = fit.get_distributions() if all_dist else fit.get_common_distributions())
    fitter.fit()
    dist = fitter.get_best(method = 'sumsquare_error')

    if plot: fitter.summary()
    if log: log_file.write((' all' if all_dist else ' common') + ' (' + datetime.datetime.now().strftime('%d.%m %H:%M') + '):\n' + str(dist) + '\n\n')

    return dist

# Generate dataframe full of possible issue timings
# Logs datapoints count and dataframe summary 
def generate(data, dist, repeats = 10000):
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
def est_effort(data: pd.DataFrame):
    eff_m = data['Sum'].quantile(.5)
    eff_o = data['Sum'].quantile(low)
    eff_p = data['Sum'].quantile(high)

    if log: log_file.write('Estimation results:\nT50' + str(eff_m) +  
                                            + '\nT' + str(round(low*100)) + ': ' + str(eff_o)
                                            + '\nT' + str(round(high*100)) + ': ' + str(eff_p))

    return eff_m, eff_p, eff_o
   
def dur_time(eff_m, eff_p, eff_o):
    # Since effort is calculated as man-seconds
    resource_sec = resource*60
    dur_o = eff_o/resource_sec
    dur_m = eff_m/resource_sec
    dur_p = eff_p/resource_sec

    if log: log_file.write('Duration results:\nD50' + str(dur_m) +  
                                            + '\nD' + str(round(low*100)) + ': ' + str(round(dur_o, 2))
                                            + '\nD' + str(round(high*100)) + ': ' + str(round(dur_p, 2)))

    return dur_m, dur_p, dur_o

# Save plots
def save_plot(path):
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def main():
    hist, new = collect_data()
    dist = found_dist if found_dist else find_distribution(hist)
    mc = generate(new, dist, repeats)
    estimation = est_effort(mc)
    log_file.close()
    if plot: save_plot(plot_file)
    
main()