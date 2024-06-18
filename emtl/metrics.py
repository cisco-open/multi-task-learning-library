# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
from math import log, sqrt

import torch


def normalized_entropy(x: torch.Tensor, nbins: int = 0) -> float:
    '''
    Compute the average entropy normalized in [0,1] for a tensor.
    Entropy is for a probability mass function, or for a sample from a discrete population, as the
    confusion in the distribution/sample. As tensors of neuron activations are continuous, we first
    need to draw a p.m.f. from the values, and do so via binning. Then, we normalize the entropy 
    value by the natural log of the number of bins. As the input is a 2D matrix, where rows are features
    and columns are observations, we compute the entropy for each row, and then take the average.
    
    Timing the function with a 2048 x 100K tensor, it takes 1.3 s to run on CPU, incompatible w/ GPU.

    Args:
        x (torch.Tensor): 2D tensor with feature rows and observation columns (D x n)
        nbins (int): _description_

    Returns:
        float: average entropy value in range [0,1]
    '''
    # input tensor has "d" features of "n" samples
    d, n = x.shape

    if nbins == 0:
        nbins = int(sqrt(n))

    # count how many values fall in each bin, stack them vertically (to make a [D x nbins] tensor)
    counts = torch.vstack([values.histogram(nbins)[0] for values in x])
    
    # build a set of D p.m.f. through a categorical distribution
    pmfs = torch.distributions.categorical.Categorical(probs=counts)

    # # compute and normalize the entropy
    entropies = pmfs.entropy()
    normalized_entropies = entropies / log(nbins)
    average_entropy = sum(normalized_entropies) / d

    return average_entropy.item()


def average_pearson_product_moment_correlation_coefficient(x: torch.Tensor) -> float:
    '''
    Compute the average "r" normalized in [0,1].
    Pearson correlation is defined between each pair of variables. As we have D variables, we will
    calculate a [D x D] matrix. Given this matrix, we want the average value, and do not care about
    positive vs negative correlation. So, we take the follwing steps:
        1. Take the absolute value of each value in the matrix
        2. Sum all the values, and subtract D (as we have D 1's in the matrix diagonal to ignore)
        3. Divide the sum by D*D-D (# of total cells - # of diagonal cells)
    
    Timing the function with a 2048 x 100K tensor, it takes 3.3 s to run on CPU, 70 ms on GPU.

    Args:
        x (torch.Tensor): [d x n] tensor where rows are variables and columns are observations.

    Returns:
        float: the average pearson product moment correlation coefficient
    '''
    d = x.shape[0]

    # x is a [D x n] matrix, so we expect the correlation matrix to be [D x D]
    M = torch.corrcoef(x)

    # normalization and averaging
    M_non_negative = abs(M)
    M_sum_non_identity_cells = M_non_negative.sum() - d
    M_cnt_non_identity_cells = (d * d) - d
    r = M_sum_non_identity_cells / M_cnt_non_identity_cells

    return r.item()


def top_k_acc(k, pred, true):
    # get indices of k highest values along last axis
    kbest = pred.argsort(-1)[:,-k:]

    # find any matches along last axis (expanding the labels to match the shape)
    bool_matches = torch.eq(true[:, None], kbest).any(dim=-1)

    # return the mean
    return bool_matches.float().mean().item()

def relative_error(pred, true):
    return abs(pred - true) / true

def multilabel_binary_classification_accuracy(pred, true):
    return (pred.round().bool() == true.bool()).float().mean()
