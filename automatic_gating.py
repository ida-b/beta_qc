import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import FlowCal
from fcswrite import write_fcs
import umap
import pynndescent
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture

# load training data
s_a1_fox = FlowCal.io.FCSData('2023-06-27/DE Ida FOXA2  A1.0014.fcs')
s_a1_fox = s_a1_fox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_a2_fox = FlowCal.io.FCSData('2023-06-27/DE Ida FOXA2  A2.0015.fcs')
s_a2_fox = s_a2_fox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_a3_fox = FlowCal.io.FCSData('2023-06-27/DE Ida FOXA2  A3.0016.fcs')
s_a3_fox = s_a3_fox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]

s_b1_fox = FlowCal.io.FCSData('2023-06-27/DE Ida FOXA2  B1.0006.fcs')
s_b1_fox = s_b1_fox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_b2_fox = FlowCal.io.FCSData('2023-06-27/DE Ida FOXA2  B2.0007.fcs')
s_b2_fox = s_b2_fox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_b3_fox = FlowCal.io.FCSData('2023-06-27/DE Ida FOXA2  B3.0008.fcs')
s_b3_fox = s_b3_fox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]

s_c1_fox = FlowCal.io.FCSData('2023-06-27/DE Ida FOXA2  C1.0010.fcs')
s_c1_fox = s_c1_fox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_c2_fox = FlowCal.io.FCSData('2023-06-27/DE Ida FOXA2  C2.0011.fcs')
s_c2_fox = s_c2_fox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_c3_fox = FlowCal.io.FCSData('2023-06-27/DE Ida FOXA2  C3.0012.fcs')
s_c3_fox = s_c3_fox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]

s_a1_sox = FlowCal.io.FCSData('2023-06-27/DE Ida SOX17 A1.0037.fcs')
s_a1_sox = s_a1_sox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_a2_sox = FlowCal.io.FCSData('2023-06-27/DE Ida SOX17 A2.0038.fcs')
s_a2_sox = s_a2_sox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_a3_sox = FlowCal.io.FCSData('2023-06-27/live (DE Ida SOX17 A3.0039).fcs')
s_a3_sox = s_a3_sox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]

s_b1_sox = FlowCal.io.FCSData('2023-06-27/DE Ida SOX17 B1.0029.fcs')
s_b1_sox = s_b1_sox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_b2_sox = FlowCal.io.FCSData('2023-06-27/DE Ida SOX17 B2.0030.fcs')
s_b2_sox = s_b2_sox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_b3_sox = FlowCal.io.FCSData('2023-06-27/DE Ida SOX17 B3.0031.fcs')
s_b3_sox = s_b3_sox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]

s_c1_sox = FlowCal.io.FCSData('2023-06-27/DE Ida SOX17 C1.0033.fcs')
s_c1_sox = s_c1_sox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_c2_sox = FlowCal.io.FCSData('2023-06-27/DE Ida SOX17 C2.0034.fcs')
s_c2_sox = s_c2_sox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
s_c3_sox = FlowCal.io.FCSData('2023-06-27/DE Ida SOX17 C3.0035.fcs')
s_c3_sox = s_c3_sox[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]

s_a1_oct = FlowCal.io.FCSData('2023-06-27/DE Ida OCT4 A1.0025.fcs')
s_a1_oct = s_a1_oct[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL13-A','FL13-W']]
s_a2_oct = FlowCal.io.FCSData('2023-06-27/DE Ida OCT4 A2.0026.fcs')
s_a2_oct = s_a2_oct[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL13-A','FL13-W']]
s_a3_oct = FlowCal.io.FCSData('2023-06-27/DE Ida OCT4 A3.0027.fcs')
s_a3_oct = s_a3_oct[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL13-A','FL13-W']]

s_b1_oct = FlowCal.io.FCSData('2023-06-27/DE Ida OCT4 B1.0018.fcs')
s_b1_oct = s_b1_oct[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL13-A','FL13-W']]
s_b2_oct = FlowCal.io.FCSData('2023-06-27/DE Ida OCT4 B2.0019.fcs')
s_b2_oct = s_b2_oct[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL13-A','FL13-W']]

s_c1_oct = FlowCal.io.FCSData('2023-06-27/DE Ida OCT4 C1.0021.fcs')
s_c1_oct = s_c1_oct[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL13-A','FL13-W']]
s_c2_oct = FlowCal.io.FCSData('2023-06-27/DE Ida OCT4 C2.0022.fcs')
s_c2_oct = s_c2_oct[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL13-A','FL13-W']]
s_c3_oct = FlowCal.io.FCSData('2023-06-27/DE Ida OCT4 C3.0023.fcs')
s_c3_oct = s_c3_oct[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL13-A','FL13-W']]

# Concatenate all samples into one array
s_fox = np.array([s_a1_fox,s_a2_fox,s_a3_fox,s_b1_fox,s_b2_fox,s_b3_fox,s_c1_fox,s_c2_fox,s_c3_fox])
s_sox = np.array([s_a1_sox,s_a2_sox,s_a3_sox,s_b1_sox,s_b2_sox,s_b3_sox,s_c1_sox,s_c2_sox,s_c3_sox])
s_oct = np.array([s_a1_oct,s_a2_oct,s_a3_oct,s_b1_oct,s_b2_oct,s_c1_oct,s_c2_oct,s_c3_oct])

# # Scale resulting batches
# scaled_batch_fox = StandardScaler().fit_transform(np.concatenate(s_fox,axis=0))
# scaled_batch_sox = StandardScaler().fit_transform(np.concatenate(s_sox,axis=0))
# scaled_batch_oct = StandardScaler().fit_transform(np.concatenate(s_oct,axis=0))

# # train models
# index_fox = pynndescent.NNDescent(
#     scaled_batch_fox,
#     n_neighbors=50,
#     diversify_prob=0.0,
#     pruning_degree_multiplier=3.0
# )
# index_fox.prepare()

# index_sox = pynndescent.NNDescent(
#     scaled_batch_sox,
#     n_neighbors=50,
#     diversify_prob=0.0,
#     pruning_degree_multiplier=3.0
# )
# index_sox.prepare()

# index_oct = pynndescent.NNDescent(
#     scaled_batch_oct,
#     n_neighbors=50,
#     diversify_prob=0.0,
#     pruning_degree_multiplier=3.0
# )
# index_oct.prepare()

# # Save the model to a file
# filename = 'models/pynndesc_model_fox.pkl'
# with open(filename, 'wb') as file:
#     pickle.dump(index_fox, file)

# filename = 'models/pynndesc_model_sox.pkl'
# with open(filename, 'wb') as file:
#     pickle.dump(index_sox, file)

# filename = 'models/pynndesc_model_oct.pkl'
# with open(filename, 'wb') as file:
#     pickle.dump(index_oct, file)

# load the model from disk
filename = 'models/pynndesc_model_fox.pkl'
with open(filename, 'rb') as file:
    index_fox = pickle.load(file)

filename = 'models/pynndesc_model_sox.pkl'
with open(filename, 'rb') as file:
    index_sox = pickle.load(file)

filename = 'models/pynndesc_model_oct.pkl'
with open(filename, 'rb') as file:
    index_oct = pickle.load(file)

# compute mean and std of each feature of s_fox, s_sox, s_oct
# used for scaling of test data
s_fox_mean = np.mean(np.concatenate(s_fox,axis=0),axis=0)
s_fox_std = np.std(np.concatenate(s_fox,axis=0),axis=0)
s_sox_mean = np.mean(np.concatenate(s_sox,axis=0),axis=0)
s_sox_std = np.std(np.concatenate(s_sox,axis=0),axis=0)
s_oct_mean = np.mean(np.concatenate(s_oct,axis=0),axis=0)
s_oct_std = np.std(np.concatenate(s_oct,axis=0),axis=0)

# load labels of training batches
labels_fox = np.load('labels_ABC/labels_fox.npy')
labels_sox = np.load('labels_ABC/labels_sox.npy')
labels_oct = np.load('labels_ABC/labels_oct.npy')

# # validation: check, whether length of labels is equal to length of data
# assert len(labels_fox) == len(scaled_batch_fox)
# assert len(labels_sox) == len(scaled_batch_sox)
# assert len(labels_oct) == len(scaled_batch_oct)

# for file in folder 'fox_files'
# load file
path = 'temp'
files = os.listdir(path)

# # create pandas dataframe to store results
# df = pd.DataFrame(columns=['file','#cells','high'],index=range(len(files)))
# # insert filenames into dataframe
# df['file'] = files

# # 2 plots with 5x6 subplots each
# fig1, axes1 = plt.subplots(5,6,figsize=(16,9))
# fig2, axes2 = plt.subplots(5,6,figsize=(16,9))

for count, file in enumerate(files):
    print('Analyzing ' + file + '...')

    # path to image
    filename = os.path.join(path, file)
    test_data = FlowCal.io.FCSData(filename)

    # ask user for which protein to test
    protein = 'oct'
    # If protein is not in (sox, fox, oct), ask again
    while protein not in ['sox','fox','oct']:
        protein = input('Incorrect input. Please enter the protein to test (choose from sox, fox or oct): ')
    # If protein is fox or sox, filter by channels ['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']
    if protein in ['fox','sox']:
        test_data = test_data[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL7-A','FL7-W','FL12-A','FL12-W']]
    # If protein is oct, filter by channels ['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL13-A','FL13-W']
    elif protein == 'oct':
        test_data = test_data[:,['FSC-A','FSC-H','FSC-W','SSC-A','SSC-W','FL1-A','FL1-W','FL13-A','FL13-W']]

    # scale test data
    if protein == 'fox':
        scaled_test_data = (test_data - s_fox_mean) / s_fox_std
    elif protein == 'sox':
        scaled_test_data = (test_data - s_sox_mean) / s_sox_std
    elif protein == 'oct':
        scaled_test_data = (test_data - s_oct_mean) / s_oct_std

    # predict labels of test data
    if protein == 'fox':
        neighbors, distances = index_fox.query(scaled_test_data, k=10, epsilon=0.2)
        # get labels of neighbors
        pred_labels = np.zeros(neighbors.shape[0], dtype=int)
        for i in range(neighbors.shape[0]):
            pred_labels[i] = np.argmax(np.bincount(labels_fox[neighbors[i,:]]))
    elif protein == 'sox':
        neighbors, distances = index_sox.query(scaled_test_data, k=10, epsilon=0.2)
        # get labels of neighbors
        pred_labels = np.zeros(neighbors.shape[0], dtype=int)
        for i in range(neighbors.shape[0]):
            pred_labels[i] = np.argmax(np.bincount(labels_sox[neighbors[i,:]]))
    elif protein == 'oct':
        neighbors, distances = index_oct.query(scaled_test_data, k=10, epsilon=0.2)
        # get labels of neighbors
        pred_labels = np.zeros(neighbors.shape[0], dtype=int)
        for i in range(neighbors.shape[0]):
            pred_labels[i] = np.argmax(np.bincount(labels_oct[neighbors[i,:]]))

    # filter test data by singlets and relevant channels
    # i.e., FL12-A and FL7-A for fox and sox, FL1-A and FL13-A for oct
    cond1 = pred_labels == 3
    cond2 = pred_labels == 4
    cond = np.logical_or(cond1, cond2)

    # # save test data filtered by singlets and relevant channels
    # write_fcs(filename='pred_singlets_sox/' + file + '_singlets.fcs', 
    #           chn_names=list(test_data.channels), data=test_data[cond,:])

    if protein == 'fox' or protein == 'sox':
        test_data_filtered = test_data[cond,:][:,[9,7]]
    elif protein == 'oct':
        test_data_filtered = test_data[cond,:][:,[5,7]]

    # # add constant such that all values in FL7 (fox, sox) / FL13 (oct) are above 1
    # test_data_filtered[:,1] += 1 - np.min(test_data_filtered[:,1])
    # # transform data to log scale
    # test_data_filtered = np.log10(test_data_filtered)

    # set all values in FL7 (fox, sox) / FL13 (oct) below 0 to 0
    test_data_filtered[:,1][test_data_filtered[:,1] < 0] = 0
    test_data_filtered[:,0] += 1 - np.min(test_data_filtered[:,0])
    # perform log+1 transformation
    test_data_filtered[:,1] = np.log10(test_data_filtered[:,1] + 1)
    test_data_filtered[:,0] = np.log10(test_data_filtered[:,0])

    # predict labels of filtered test data
    gmm = GaussianMixture(n_components=2).fit(test_data_filtered[:,1].reshape(-1, 1))
    labels = gmm.predict(test_data_filtered[:,1].reshape(-1, 1))
    # print(np.unique(labels, return_counts=True))

    # set label 3 to low expressing cells, label 4 to high expressing cells
    idx = np.argmax(gmm.means_)
    pred_labels[cond] = np.where(labels == idx, 4, 3)
    # print(np.unique(pred_labels, return_counts=True))

    # # save predicted labels
    # np.save('pred_labels_fox/' + file + '_pred_labels.npy', pred_labels)

    # calculate ratio of labels 
    idx = np.argmax(gmm.means_)
    ratio = np.sum(labels == idx) / len(labels)
    # print('Ratio of high to low expressing cells in test data: {}'.format(ratio))
    # # store ratio in dataframe
    # df.loc[df['file'] == file, 'high'] = np.round(ratio,2)
    # df.loc[df['file'] == file, '#cells'] = len(labels)

    # # divide count by 6 and round down to nearest integer to get row index, use count%6 to get column index
    # axes2[count//6, count%6].scatter(test_data_filtered[:,0], test_data_filtered[:,1], c=labels, cmap='Spectral', s=1)
    # axes2[count//6, count%6].set_ylabel('FL7-A' if protein == 'fox' or protein == 'sox' else 'FL13-A')
    # axes2[count//6, count%6].set_xlabel('FL12-A' if protein == 'fox' or protein == 'sox' else 'FL1-A')
    # axes2[count//6, count%6].set_title(file.split(' DE')[0] + f': {ratio:.2f}')

    # num_bins = 100
    # # create histogram of FL7 (fox, sox) / FL13 (oct) values where the bars are colored by the majority label
    # n, bins, patches = axes1[count//6, count%6].hist(test_data_filtered[:,1], bins=num_bins, 
    #                                                 range=(np.min(test_data_filtered[:,1]), 
    #                                                        np.max(test_data_filtered[:,1])))

    num_bins = 100
    # create histogram of FL7 (fox, sox) / FL13 (oct) values where the bars are colored by the majority label
    n, bins, patches = plt.hist(test_data_filtered[:,1], bins=num_bins, 
                                                    range=(np.min(test_data_filtered[:,1]), 
                                                           np.max(test_data_filtered[:,1])))
    
    # for i in range(num_bins):
    #     bin_values = labels[(test_data_filtered[:,1] >= bins[i]) & (test_data_filtered[:,1] < bins[i+1])]
    #     if len(bin_values) > 0:
    #         bin_labels, bin_counts = np.unique(bin_values, return_counts=True)
    #         majority_label = bin_labels[np.argmax(bin_counts)]
    #         # choose colors from colormap Spectral
    #         color = plt.cm.Spectral(majority_label / 1) 
    #         patches[i].set_facecolor(color)

    # axes1[count//6, count%6].set_xlabel('FL7-A' if protein == 'fox' or protein == 'sox' else 'FL13-A')
    # axes1[count//6, count%6].set_title(file.split(' DE')[0] + f': {ratio:.2f}')

    for i in range(num_bins):
        bin_values = labels[(test_data_filtered[:,1] >= bins[i]) & (test_data_filtered[:,1] < bins[i+1])]
        if len(bin_values) > 0:
            bin_labels, bin_counts = np.unique(bin_values, return_counts=True)
            majority_label = bin_labels[np.argmax(bin_counts)]
            # choose colors from colormap Spectral
            color = plt.cm.Spectral(majority_label / 1) 
            patches[i].set_facecolor(color)

    plt.xlabel('FL7-A' if protein == 'fox' or protein == 'sox' else 'FL13-A')
    plt.title(file.split(' DE')[0] + ' SOX17 expression: ' + f'{ratio:.2f}')
    plt.show()
    
# # Adjust spacing between subplots
# fig1.tight_layout()
# fig2.tight_layout()

# # save figures
# fig1.savefig('oct_histograms.png')
# fig2.savefig('oct_scatterplots.png')

# # show figures
# plt.show()