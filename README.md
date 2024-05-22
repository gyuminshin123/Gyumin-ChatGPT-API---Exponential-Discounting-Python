#Start with import of all the necessary code

import os
import numpy as np
import matplotlib
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, optimize
from pandas import DataFrame, Series
import seaborn as sns
import random as rd
import scipy.stats
import multiprocessing
from scipy.stats import norm
import statsmodels.stats.moment_helpers
from scipy.stats import beta
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os.path as op
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#Define that we are using exponential functions.
#A stands for amount
#d = delay amount of days (x-values)
#k = parameter

def exponential(A,d,k):
    return A * 1.0/(1.0+k*d)

def get_LL(params,sub_df):
    k,m = params
    LL = 0
    for i in range(sub_df.shape[0]):
        SV_SS = exponential(sub_df.loc[i,'amountSS'],
                            sub_df.loc[i,'delaySS'],
                            k)
        SV_LL = exponential(sub_df.loc[i,'amountLL'],
                            sub_df.loc[i,'delayLL'],
                            k)

        p = 1 / (1 + np.exp(-1*m*(SV_SS-SV_LL)) )

        if sub_df.loc[i,'choice'] == 'LL':
            p = 1-p
        LL += np.log(p)
    return -1*LL


#arg = (data) is where we want to put our CSV file

data = df = pd.read_csv(r'C:/Users/shing/OneDrive - email.ucr.edu/Desktop/API Chat-GPT/Decision questions.csv')
res = minimize(get_LL, [-4.5, -1], 
               args = (data),
               method = 'BFGS')

