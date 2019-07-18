import cv2
import yaml
import os
import pydicom
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from numpy import log,exp,sqrt,pi,arcsinh

man = "BAI_TIESHAN"
pic = "BAI_TIESHAN.CT.RT_0_PETCT_WHOLEBODY_PUMC_(ADULT).0003.0184.2015.08.13.10.46.34.359375.1202728701.IMA"
pos = 0    # 0 for 'up', 1 for 'down'

if(pos == 0):
    mu, lambda1, lambda2 = 0.01, 5.0, 18.0
else:
    mu, lambda1, lambda2 = 0.01, 1.0, 4.0

dt, tol = 0.1, 0.00000001
max_iter = 5000
init_level_set = 'small disk'
extended_output = False

KWs = {
    'bone':{
        'a':2.6217981849701704,
        'b':0.892985122362825,
        'loc':2583.64530273106,
        'scale':31.325450292326998
    },
    'spine':{
        'a':-1.9368975025645425,
        'b':0.6575388750887041,
        'loc':987.4907834755124,
        'scale':15.930377780724314
    }
}

PIVOT = {
    'spine':1008.67,
    'bone':2495.79,
}

def start():
    os.system("python chan_vese.py")
    os.system("python fit.py")

def read_dicom(fpath):
    img_im = pydicom.read_file(fpath)
    img = img_im.pixel_array.astype(np.float)
    return img

def read_age(fpath):
    meta_data = pydicom.dcmread(fpath)
    return meta_data.PatientAge

def fit_johnsonsu(data,bins=200,start=0,end=4000,size=8000):

    distribution = st.johnsonsu
    #params = distribution.fit(data,fa=-5,fb=2.5,floc=850)
    params = distribution.fit(data)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    x = np.linspace(start,end,size)
    y = distribution.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y,x)
    sse = np.sum(np.power(y - pdf, 2.0))

    return (params,pdf,sse)

def dichotomy(left,right,delta,a,b,loc,scale):
    l,s = loc,scale
    f = lambda x:-((b*exp((-1/2)*(a+b*arcsinh((x-l)/s))**2)*\
                    ( a*b*s*sqrt((l**2-2*l*x+s**2+x**2)/s**2)+b**2*s*sqrt((l**2-2*l*x+s**2+x**2)/s**2)*arcsinh((x-l)/s)-l+x))/\
                      sqrt(2*pi)*s**3*(l**2-2*l*x+s**2+x**2)**(3/2))
    while(right-left>delta):
        mid = (left+right)/2
        if(f(mid)>0):
            left = mid
        else:
            right = mid
    return left

def gen_f(a,b,loc,scale):
    l,s = loc,scale
    return lambda x:(b/(s*sqrt(2*pi)))*(1/sqrt(1+((x-l)/s)**2))*exp((-1/2)*(a+b*arcsinh((x-l)/s))**2)

def predraw():
    for key,kw in KWs.items():
        loc = kw['loc']
        scale = kw['scale']
        arg = [kw['a'],kw['b']]

        x = np.linspace(0,4000,8000)
        y = st.johnsonsu.pdf(x,**kw)
        pdf = pd.Series(y,x)

        pivot = dichotomy(0,4000,0.1,**kw)
        height = gen_f(**kw)(pivot)
        plt.vlines(pivot,0,height)

        ax = pdf.plot(lw=1, legend=False)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)

def draw_johnsonsu(data,kw,pdf,bins=30,show = True,fpath=None):
    plt.figure(figsize=(12,8))
    plt.ylim(ymax=0.008)

    predraw()

    pivot = dichotomy(0,4000,0.1,**kw)
    height = gen_f(**kw)(pivot)
    plt.vlines(pivot,0,height,colors='green')

    ax = pdf.plot(lw=1, legend=True,label='{:.2f}'.format(pivot))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
    plt.hist(data, bins=bins, density=True, alpha=0.5,color='lightblue')

    if fpath:
        plt.savefig(fpath)
    ''' 
    if show:
        plt.show()
    '''
    plt.close('all')
    return pivot


def draw_all(data, kw, pdf, bins=30, show=True, fpath=None):
    plt.figure(figsize=(12, 8))
    plt.ylim(ymax=0.008)

    predraw()

    pivot = dichotomy(0, 4000, 0.1, **kw)
    height = gen_f(**kw)(pivot)
    plt.vlines(pivot, 0, height, colors='green')

    ax = pdf.plot(lw=1, legend=True, label='{:.2f}'.format(pivot))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
    plt.hist(data, bins=bins, density=True, alpha=0.5, color='lightblue')

    if show:
        plt.show()
    if fpath:
        plt.savefig(fpath)
    plt.close('all')


if __name__ == '__main__':
    start()