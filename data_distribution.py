# data distribution

""" 
d1 = df1["1m_close_future_pct"]

xs = np.arange(d1.min(), d1.max(), 0.1)
fit = stats.norm.pdf(xs, np.mean(d1), np.std(d1))
plt.plot(xs, fit, label='Normal Dist.', lw=3)
plt.hist(d1, 50, density=True, label='Actual Data');
plt.legend();

output = graph
"""

"""

# Run normal test on the data
stat, p_val = normaltest(d1)
# Check the p-value of the normaltest
print('\nNormaltest p-value is: {:1.2f} \n'.format(p_val))

# With alpha value of 0.05, how should we proceed
check_p_val(p_val, alpha=0.05)

print ()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import normaltest

"""
Advance with argparse

input = file_name(str), column_name(str), alpha(float) 

"""

def main():
    parser = argparse.ArgumentParser(description='Calculate volume of a Cylinder')
    # parser.add_argument('radius',type=int,help='Radius of a Cylinder')
    # parser.add_argument('height',type=int,help='Height of a Cylinder')
    # add -r , -H fpr specific the position 
    # add metavar, required 
    parser.add_argument('-r','--radius',type=int,metavar='',required=True,help='Radius of a Cylinder')
    parser.add_argument('-H','--height',type=int,metavar='',required=True,help='Height of a Cylinder')
    args = parser.parse_args()
    df = pd.read_csv('/Users/yothinpu/Documents/CS50P_2022/test_data_19.csv')    
    df = df["1m_close_future_pct"]
    run_test(df)
    y = plot_distribution(df)

def plot_distribution(data):
    # define the column in dataFrame
    # data = df['return']
    xs = np.arange(data.min(), data.max(), 0.1)
    fit = stats.norm.pdf(xs, np.mean(data), np.std(data))
    plt.plot(xs, fit, label='Normal Dist.', lw=3)
    plt.hist(data, 50, density=True, label='Actual Data');
    plt.legend()
    plt.show()


def run_test(data):
    # Run normal test on the data
    stat, p_val = normaltest(data)
    # Check the p-value of the normaltest
    print('\nNormaltest p-value is: {:1.2f} \n'.format(p_val))
    # With alpha value of 0.05, how should we proceed
    return check_p_val(p_val, alpha=0.05)

def check_p_val(p_val, alpha):
    if p_val < alpha:
        print('We have evidence to reject the null hypothesis.')
    else:
        print('We do not have evidence to reject the null hypothesis.')
    return p_val < alpha

if __name__ == "__main__":
    main()
