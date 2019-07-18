import os
import json
import yaml
import pandas as pd
from tools import fit_johnsonsu,draw_johnsonsu, read_age
from tools import man, pic, pos

'''
# work for batch.py
man, pic, pos= '', '', ''
def fit_batch(_man, _pic, _pos):
    global man, pic, pos
    man, pic, pos = _man, _pic, _pos
    start()
'''

def start():
    bright_dir = os.path.join('brightness', man)
    if(pos == 0):
        fpath = os.path.join(bright_dir, 'UP_' + pic.replace(".IMA", ".json"))
        name = 'UP_' + os.path.splitext(pic)[0]
    else:
        fpath = os.path.join(bright_dir, 'DOWN_' + pic.replace(".IMA", ".json"))
        name = 'DOWN_' + os.path.splitext(pic)[0]
    with open(fpath, 'r') as f:
        info = json.load(f)
    if(pos == 0):
        data = info['up']
    else:
        data = info['down']
    params, pdf, sse = fit_johnsonsu(data)
    a, b, loc, scale = params
    kw = {'a': a, 'b': b, 'loc': loc, 'scale': scale}
    params_fit = {k: float(v) for k, v in kw.items()}
    save_path = os.path.join('figures_brightness', man, name + '.png')
    params_fit['Extremum'] = draw_johnsonsu(data, kw, pdf, show=True, fpath=save_path)
    age = str(read_age(os.path.join("lumbar_data", man, pic)))
    if(age[0] == '0' and age[1] == '0'):
        age_str = age[2]
    elif(age[0] == '0'):
        age_str = age[1:3]
    else:
        age_str = age[0:3]
    params_fit['Age'] = float(age_str)

    upath = os.path.join('params_fit', man, name + '.yaml')
    with open(upath, 'w') as f:
        yaml.dump(params_fit, f, default_flow_style=False)


if __name__ == '__main__':
    start()