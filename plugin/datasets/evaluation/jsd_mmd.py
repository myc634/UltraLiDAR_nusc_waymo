
import numpy as np
import concurrent.futures
from functools import partial
from scipy.linalg import toeplitz
import torch.distributions as dist
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm, trange

    
def jsd_2d(p, q):
    from scipy.spatial.distance import jensenshannon
    return jensenshannon(p.flatten(), q.flatten())


def kernel_parallel_unpacked(x, samples2, kernel):
  d = 0
  for s2 in samples2:
    d += kernel(x, s2)
  return d


def kernel_parallel_worker(t):
  return kernel_parallel_unpacked(*t)

def gaussian(x, y, sigma=0.5):  
  support_size = max(len(x), len(y))
  # convert histogram values x and y to float, and make them equal len
  x = x.astype(np.float)
  y = y.astype(np.float)
  if len(x) < len(y):
    x = np.hstack((x, [0.0] * (support_size - len(x))))
  elif len(y) < len(x):
    y = np.hstack((y, [0.0] * (support_size - len(y))))


  import time
  #TODO: Calculate empirical sigma by fitting dist to gaussian 

  dist = np.linalg.norm(x - y, 2)
  # dist_test = np.norm()
  res = np.exp(-dist * dist / (2 * sigma * sigma))

  return res


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
  ''' Discrepancy between 2 samples '''
  d = 0

  if not is_parallel:
    for s1 in tqdm(samples1):
      for s2 in samples2:

        d += kernel(s1, s2, *args, **kwargs)
  else:
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for dist in executor.map(kernel_parallel_worker, [
    #       (s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1
    #   ]):
    #     d += dist

    with concurrent.futures.ThreadPoolExecutor() as executor:
      for dist in executor.map(kernel_parallel_worker, [
          (s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1
      ]):
        d += dist

  d /= len(samples1) * len(samples2)

  return d

def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
  ''' MMD between two samples '''
  # normalize histograms into pmf  
  # breakpoint()
  if is_hist:
    samples1 = [s1 / np.sum(s1) for s1 in samples1]
    samples2 = [s2 / np.sum(s2) for s2 in samples2]
  s1 = disc(samples1, samples1, kernel, *args, **kwargs)
  print('===============================')
  print('s1: ', s1)
  s2 = disc(samples2, samples2, kernel, *args, **kwargs)
  print('--------------------------')
  print('s2: ', s2)
  print('--------------------------')

  cross = disc(samples1, samples2, kernel, *args, **kwargs)
  print('cross: ', cross)
  print('===============================')
  return s1 + s2 - 2 * cross
          
def point_cloud_to_histogram(field_size, bins, point_cloud):

    point_cloud_flat = point_cloud[:,:2].detach().cpu().numpy()

    square_size = field_size / bins

    halfway_offset = 0
    if(bins % 2 == 0):
        halfway_offset = (bins / 2) * square_size
    else:
        print('ERROR')

    histogram = np.histogramdd(point_cloud_flat, bins=bins, range=([-halfway_offset, halfway_offset], [-halfway_offset, halfway_offset]))

    return histogram