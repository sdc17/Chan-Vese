import os
import yaml
from chan_vese import chan_vese_batch
from fit import fit_batch

def start():
    global man, pic, pos, lambda1, lambda2, mu
    cnt = 0
    dir1 = os.listdir('params_split')
    for i in dir1:
        man = i
        path = os.path.join('params_split', i)
        dir2 = os.listdir(path)
        for j in dir2:
            if (str(j)[0] == 'U'):
                pic = j.replace(".yaml", ".IMA")[3:]
            else:
                pic = j.replace(".yaml", ".IMA")[5:]
            fpath = os.path.join('params_split', i, j)
            with open(fpath, 'r') as f:
                data = yaml.load(f)
                mu = data['mu']
                lambda1 = data['lambda1']
                lambda2 = data['lambda2']
                if (str(j)[0] == 'U'):
                    pos = 0
                else:
                    pos = 1

            chan_vese_batch(man, pic, pos, lambda1, lambda2, mu)
            fit_batch(man, pic, pos)
            cnt += 1
            print(cnt)

if __name__ == '__main__':
    start()