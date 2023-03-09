#!/usr/bin/env python3

""" data_loader
(1) vcf2hap: Convert VCF files
    Convert VCF to .gen .hap and .legend.
    Modified from SNP_GAIN/snp_sampler_rand
(2) smart_open
(3) data_loader
(4) maf_loader
"""
import os
import gzip
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder


def vcf2hap(inFile):
    # Init Lookup dict
    gt_to_hap = {
        '0|0': [0, 0],
        '0|1': [0, 1],
        '1|0': [1, 0],
        '1|1': [1, 1]}

    # Init index
    chrom_idx = -1
    pos_idx = -1
    id_idx = -1
    ref_idx = -1
    alt_idx = -1
    format_idx = -1
    snp_cnt = 0

    # Init list
    samples = []
    pos = []
    outLeg = []
    outHap = []

    # Get Header idx
    with smart_open(inFile, 'rt') as f:
        for line in f:
            if line.startswith("#CHROM"):
                line = line.split("\t")

                chrom_idx = line.index('#CHROM')
                pos_idx = line.index('POS')
                id_idx = line.index('ID')
                ref_idx = line.index('REF')
                alt_idx = line.index('ALT')
                format_idx = line.index('FORMAT')

                samples = line[format_idx + 1:]

    # Parse Loop
    with smart_open(inFile, 'rt') as f:
        for line in f:
            if not line.startswith("#"):
                snp_cnt += 1
                line = line.strip()
                line = line.split("\t")

                # write legend constant column
                outLeg.append(
                    [line[id_idx],
                     line[pos_idx],
                     line[ref_idx],
                     line[alt_idx]])

                pos.append(line[pos_idx])

                # convert gt to hap
                gt_hap = []
                smaple_cnt = len(line[format_idx + 1:])
                for i in range(smaple_cnt):
                    gt_vcf = line[format_idx + 1 + i]
                    gt_hap.extend(gt_to_hap[gt_vcf])
                outHap.append(gt_hap)

        outHap = np.array(outHap)
    return samples, pos, outHap


def smart_open(filename, mode="r"):
    if filename.endswith(".gz"):
        return(gzip.open(filename, mode))
    else:
        return(open(filename, mode))


def data_loader(inFile):
    """Loads datasets.
    Load varoius genotype file format to numpy array.
    The returned data has been transposed.
    One column represent one SNPs,
    One row represent one haplotype.

    Args:
        - inFile: relative path to data

    Returns:
        data: original data
    """
    filename, file_extension = os.path.splitext(inFile)

    if file_extension == '.gz':
        filename, file_extension = os.path.splitext(filename)

    if file_extension == '.hap':
        data = smart_open(inFile)
        data = np.genfromtxt(data, delimiter=' ')
        data = data.transpose()
    elif file_extension == '.vcf':
        sample, pos, data = vcf2hap(inFile)
        data = data.transpose()
    elif file_extension == '.hapt':
        data = []
        with smart_open(inFile) as f:
            for line in f:
                line = line.strip().split(" ")
                line = line[2:]
                data.append(line)
        data = np.asarray(data, dtype=np.int32)
    elif file_extension == '.gen':
        data = []
        with smart_open(inFile) as f:
            for line in f:
                line = line.strip().split(" ")
                line = line[5:]
                data.append(line)
        data = np.asarray(data)
        data = data.transpose()
    elif file_extension == '.npy':
        data = np.load(inFile)
    else:
        raise ValueError('Not supported data format.')

    return data


def tag_target_split(data, tag_pos, all_pos):
    tag_pos = np.genfromtxt(tag_pos, delimiter='\t')
    all_pos = np.genfromtxt(all_pos, delimiter='\t')

    tag_pos = tag_pos[:, 1]
    all_pos = all_pos[:, 1]

    tag_mask = np.isin(all_pos, tag_pos)
    target_mask = np.isin(all_pos, tag_pos, invert=True)

    data_tag = data[:, tag_mask]
    data_target = data[:, target_mask]

    return data_tag, data_target


def data_processor(train_data, test_data, tag_pos, all_pos):
    # Load mnist
    if train_data == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(-1, 28 * 28)
        X_train = np.where(X_train > 127, 1, 0)
        X_test = X_test.reshape(-1, 28 * 28)
        X_test = np.where(X_test > 127, 1, 0)

        labelencoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(-1, 1)
        y_train = labelencoder.fit_transform(y_train)
        y_test = y_test.reshape(-1, 1)
        y_test = labelencoder.fit_transform(y_test)
        label_dim = 10

    # Load SNP data
    else:
        X_train = data_loader(train_data)
        X_test = data_loader(test_data)

        y_train, X_train = tag_target_split(X_train, tag_pos, all_pos)
        y_test, X_test = tag_target_split(X_test, tag_pos, all_pos)
        label_dim = y_train.shape[1]

    # Type Casting
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')

    return X_train, y_train, X_test, y_test, label_dim


def maf_loader(inFile):
    filename, file_extension = os.path.splitext(inFile)

    if file_extension == '.gz':
        filename, file_extension = os.path.splitext(filename)

    if file_extension == '.maf':
        data = smart_open(inFile)
        data = np.genfromtxt(data, delimiter=' ')
        data = data.transpose()
    else:
        raise ValueError('Not supported data format.')
    return data


def data_sampler(data_x, miss_rate, all_pos=''):

    # Parameters
    no, dim = data_x.shape

    # Random missing (0 denote missing)
    if isinstance(miss_rate, float) == True:
        data_m = np.random.uniform(0.0, 1.0, [no, dim])
        data_m = 1 * (data_m < (1 - miss_rate))

    # Array missing
    else:
        tag_pos = np.genfromtxt(miss_rate, delimiter='\t')
        all_pos = np.genfromtxt(all_pos, delimiter='\t')

        tag_pos = tag_pos[:, 1]
        all_pos = all_pos[:, 1]

        data_m = np.isin(all_pos, tag_pos)
        data_m = data_m.astype(float)

        data_m = np.tile(data_m, (no, 1))

    # Introduce missingness
    miss_data_x = data_x.copy()
    miss_data_x = miss_data_x.astype(float)
    miss_data_x[data_m == 0] = 0

    return miss_data_x, data_m


def map_loader(map_name):
    data_map = np.loadtxt(map_name, delimiter=' ', skiprows=1)
    data_map = data_map[:, 1]
    data_map = (data_map - np.min(data_map)) / (np.max(data_map) - np.min(data_map))
    data_map = np.asarray(data_map)
    data_map = data_map.transpose()
    return data_map
