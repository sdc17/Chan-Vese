import os
import yaml
import math
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from numpy import log,exp,sqrt,pi,arcsinh

def start():
    up_age = []
    up_extremum = []
    up_a = []
    up_b = []
    up_loc = []
    up_scale = []
    down_age = []
    down_extremum = []
    down_a = []
    down_b = []
    down_loc = []
    down_scale = []
    dir1 = os.listdir('params_fit')
    for i in dir1:
        path = os.path.join('params_fit', i)
        dir2 = os.listdir(path)
        for j in dir2:
            fpath = os.path.join('params_fit', i, j)
            with open(fpath, 'r') as f:
                data = yaml.load(f)
                if(math.fabs(data['a']) < 1e4 and math.fabs(data['b']) < 1e4 and math.fabs(data['loc']) < 1e4
                        and math.fabs(data['scale']) < 1e3):
                    if (str(j)[0] == 'U'):
                        up_age.append(data['Age'])
                        up_extremum.append(data['Extremum'])
                        up_a.append(data['a'])
                        up_b.append(data['b'])
                        up_loc.append(data['loc'])
                        up_scale.append(data['scale'])
                    else:
                        down_age.append(data['Age'])
                        down_extremum.append(data['Extremum'])
                        down_a.append(data['a'])
                        down_b.append(data['b'])
                        down_loc.append(data['loc'])
                        down_scale.append(data['scale'])

    # age = up_age + down_age
    # extremum = up_extremum + down_extremum

    up_age_pd = pd.Series(up_age)
    up_extremum_pd = pd.Series(up_extremum)
    up_a_pd = pd.Series(up_a)
    up_b_pd = pd.Series(up_b)
    up_loc_pd = pd.Series(up_loc)
    up_scale_pd = pd.Series(up_scale)
    down_age_pd = pd.Series(down_age)
    down_extremum_pd = pd.Series(down_extremum)
    down_a_pd = pd.Series(down_a)
    down_b_pd = pd.Series(down_b)
    down_loc_pd = pd.Series(down_loc)
    down_scale_pd = pd.Series(down_scale)
    # age_pd = pd.Series(age)
    # extremum_pd = pd.Series(extremum)
    corr_up_age_extremum = up_age_pd.corr(up_extremum_pd)
    corr_down_age_extremum = down_age_pd.corr(down_extremum_pd)
    # corr_tot = age_pd.corr(extremum_pd)
    corr_up_age_a = up_age_pd.corr(up_a_pd)
    corr_down_age_a = down_age_pd.corr(down_a_pd)
    corr_up_age_b = up_age_pd.corr(up_b_pd)
    corr_down_age_b = down_age_pd.corr(down_b_pd)
    corr_up_age_loc = up_age_pd.corr(up_loc_pd)
    corr_down_age_loc = down_age_pd.corr(down_loc_pd)
    corr_up_age_scale = up_age_pd.corr(up_scale_pd)
    corr_down_age_scale = down_age_pd.corr(down_scale_pd)

    '''
    # 3 pics
    plt.figure(figsize=(11, 7))
    bias = plt.get_current_fig_manager()
    bias.window.wm_geometry("+100+0")
    plt.subplot(2, 1, 1)
    plt.scatter(age, extremum)
    plt.title('corr :' + str(corr_tot), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("Extremum")
    plt.subplot(2, 2, 3)
    plt.scatter(up_age, up_extremum)
    plt.title('corr_up :' + str(corr_up), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("Extremum")
    plt.subplot(2, 2, 4)
    plt.scatter(down_age, down_extremum)
    plt.title('corr_down :' + str(corr_down), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("Extremum")
    '''

    '''
    # 2pics age_extremum
    plt.figure(figsize=(13, 5))
    bias = plt.get_current_fig_manager()
    bias.window.wm_geometry("+30+100")
    plt.subplot(1, 2, 1)
    plt.scatter(up_age, up_extremum)
    plt.title('corr_up_age_extremum :' + str(corr_up_age_extremum), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("Extremum")
    plt.subplot(1, 2, 2)
    plt.scatter(down_age, down_extremum)
    plt.title('corr_down_age_extremum :' + str(corr_down_age_extremum), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("Extremum")
    '''


    '''
    #2pics age_a
    plt.figure(figsize=(13, 5))
    bias = plt.get_current_fig_manager()
    bias.window.wm_geometry("+30+100")
    plt.subplot(1, 2, 1)
    plt.scatter(up_age, up_a)
    plt.title('corr_up_age_a :' + str(corr_up_age_a), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("a")
    plt.subplot(1, 2, 2)
    plt.scatter(down_age, down_a)
    plt.title('corr_down_age_a :' + str(corr_down_age_a), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("a")
    '''

    '''
    # 2pics age_b
    plt.figure(figsize=(13, 5))
    bias = plt.get_current_fig_manager()
    bias.window.wm_geometry("+30+100")
    plt.subplot(1, 2, 1)
    plt.scatter(up_age, up_b)
    plt.title('corr_up_age_b :' + str(corr_up_age_b), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("b")
    plt.subplot(1, 2, 2)
    plt.scatter(down_age, down_b)
    plt.title('corr_down_age_b :' + str(corr_down_age_b), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("b")
    '''

    '''
    # 2pics age_loc
    plt.figure(figsize=(13, 5))
    bias = plt.get_current_fig_manager()
    bias.window.wm_geometry("+30+100")
    plt.subplot(1, 2, 1)
    plt.scatter(up_age, up_loc)
    plt.title('corr_up_age_loc :' + str(corr_up_age_loc), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("loc")
    plt.subplot(1, 2, 2)
    plt.scatter(down_age, down_loc)
    plt.title('corr_down_age_loc :' + str(corr_down_age_loc), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("loc")
    '''

    '''
    # 2pics age_scale
    plt.figure(figsize=(13, 5))
    bias = plt.get_current_fig_manager()
    bias.window.wm_geometry("+30+100")
    plt.subplot(1, 2, 1)
    plt.scatter(up_age, up_scale)
    plt.title('corr_up_age_scale :' + str(corr_up_age_scale), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("scale")
    plt.subplot(1, 2, 2)
    plt.scatter(down_age, down_scale)
    plt.title('corr_down_age_scale :' + str(corr_down_age_scale), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("scale")
    '''

    #8 pics
    plt.figure(figsize=(16, 7))
    bias = plt.get_current_fig_manager()
    bias.window.wm_geometry("+0+0")

    plt.subplot(2, 4, 1)
    plt.scatter(up_age, up_a)
    plt.title('up_age_a :' + str(corr_up_age_a), fontproperties='SimHei')
    plt.ylabel("a")
    plt.subplot(2, 4, 2)
    plt.scatter(down_age, down_a)
    plt.title('down_age_a :' + str(corr_down_age_a), fontproperties='SimHei')
    plt.ylabel("a")

    plt.subplot(2, 4, 3)
    plt.scatter(up_age, up_b)
    plt.title('up_age_b :' + str(corr_up_age_b), fontproperties='SimHei')
    plt.ylabel("b")
    plt.subplot(2, 4, 4)
    plt.scatter(down_age, down_b)
    plt.title('down_age_b :' + str(corr_down_age_b), fontproperties='SimHei')
    plt.ylabel("b")

    plt.subplot(2, 4, 5)
    plt.scatter(up_age, up_loc)
    plt.title('up_age_loc :' + str(corr_up_age_loc), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("loc")
    plt.subplot(2, 4, 6)
    plt.scatter(down_age, down_loc)
    plt.title('down_age_loc :' + str(corr_down_age_loc), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("loc")

    plt.subplot(2, 4, 7)
    plt.scatter(up_age, up_scale)
    plt.title('up_age_scale :' + str(corr_up_age_scale), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("Scale")
    plt.subplot(2, 4, 8)
    plt.scatter(down_age, down_scale)
    plt.title('down_age_scale :' + str(corr_down_age_scale), fontproperties='SimHei')
    plt.xlabel("Age")
    plt.ylabel("Scale")


    # save_path = 'analysis/age_extremum.png'
    # save_path = 'analysis/age_a&&age_b&&age_loc&&age_scale.png'
    # plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    start()