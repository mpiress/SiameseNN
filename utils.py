"""!

@PyMid is devoted to manage multitask algorithms and methods.

@authors Michel Pires da Silva (michelpires@dcc.ufmg.br)

@date 2014-2018

@copyright GNU Public License

@cond GNU_PUBLIC_LICENSE
    PyMid is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    PyMid is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
@endcond

"""

import numpy as np
import pandas as pd
import math, os, csv 

from sklearn.preprocessing import normalize, MinMaxScaler
from collections import Counter
from cache_map import cache_fx


def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return pd.DataFrame(fv)
    

def get_delimiter(path):
    delimiter = csv.Sniffer()
    with open(path, 'r') as fp:
        fp.readline()
        line = fp.readline()
    try:
        d = delimiter.sniff(line, [',',';', '\t']).delimiter
    except:
        d = None
    return d
    

def read_file(path, has_header=False, cols=None, norm=False):
    key = path[path.rfind('/')+1:]
    assert os.path.isfile(path), '[ERROR] file is not found in dataset folder, read_file error'
    
    d = get_delimiter(path)
    if has_header:
        data = pd.read_csv(path, delimiter=d, usecols=cols, low_memory=False)
    else:
        data = pd.read_csv(path, header=None, delimiter=d, low_memory=False) 

    if norm:
        min_max_scaler = MinMaxScaler()
        data = pd.DataFrame(min_max_scaler.fit_transform(data.values))           
            
    return data.sample(frac=1, random_state=42)




