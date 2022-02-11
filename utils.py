import numpy as np
import math
import pandas as pd
import itertools
import torch
import scipy.sparse as sp


import pickle
import numpy as np
import math
import pandas as pd
import itertools
import torch
import scipy.sparse as sp

import torch
from utils_data import *
import pandas as pd
import collections
import numpy as np
import os
import pickle
from collections import defaultdict
import math
import progressbar
import itertools


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def generate_sequence(input_data, min_seq_len, min_seq_num):
	bar = progressbar.ProgressBar(maxval=input_data.index[-1], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	input_data['Local_sg_time'] = pd.to_datetime(input_data['Local_Time_True'])
	total_sequences_dict = {} 
	max_seq_len = 0 
	valid_visits = [] 
	bar.start()
	for user in input_data['UserId'].unique():
		user_visits = input_data[input_data['UserId'] == user]
		user_sequences = [] 
		unique_date_group = user_visits.groupby([user_visits['Local_sg_time'].dt.date]) 
		for date in unique_date_group.groups:
			single_date_visit = unique_date_group.get_group(date)
			single_sequence = _remove_consecutive_visit(single_date_visit, bar) 
			if len(single_sequence) >= min_seq_len: 
				user_sequences.append(single_sequence)
				if len(single_sequence) > max_seq_len: 
					max_seq_len = len(single_sequence) 
		if len(user_sequences) >= min_seq_num: 
			total_sequences_dict[user]=np.array(user_sequences,dtype=object)
			valid_visits = valid_visits + list(itertools.chain.from_iterable(user_sequences))
	bar.finish()
	user_reIndex_mapping = np.array(list(total_sequences_dict.keys()),dtype=object)
	return total_sequences_dict, max_seq_len, valid_visits, user_reIndex_mapping
def _remove_consecutive_visit(visit_record, bar):
	clean_sequence = []
	for index,visit in visit_record.iterrows():
		bar.update(index)
		clean_sequence.append(index)
	return clean_sequence

def aug_sequence(input_sequence_dict, min_len):
	augmented_sequence_dict, ground_truth_dict = {}, {}
	for user in input_sequence_dict.keys():
		user_sequences, ground_truth_sequence = [], []
		for seq in input_sequence_dict[user]:
			if len(seq)>min_len:
				for i in range(len(seq)-min_len+1):
					user_sequences.append(seq[0:i+min_len])
					ground_truth_sequence.append(seq[i+min_len-1:])
			else: 
				user_sequences.append(seq)
				ground_truth_sequence.append([seq[-1]])
		augmented_sequence_dict[user] = np.array(user_sequences,dtype=object)
		ground_truth_dict[user] = np.array(ground_truth_sequence,dtype=object)
	return augmented_sequence_dict, ground_truth_dict

def pad_sequence(input_sequence_dict, max_seq_len):
	padded_sequence_dict = {}
	for user in input_sequence_dict.keys():
		user_sequences = []
		for seq in input_sequence_dict[user]:
			seq = np.pad(seq, (0,max_seq_len - len(seq)), 'constant', constant_values=-1,)
			user_sequences.append(seq)
		padded_sequence_dict[user] = np.array(user_sequences,dtype=object)
	return padded_sequence_dict

def generate_POI_sequences(input_data, visit_sequence_dict):
	POI_sequences = []
	for user in visit_sequence_dict:
		user_POI_sequences = []
		for seq in visit_sequence_dict[user]:
			POI_sequence = []
			for visit in seq:
				if visit != -1:
					POI_sequence.append(input_data['Location_id'][visit])
				else: 
					POI_sequence.append(-1)
			user_POI_sequences.append(POI_sequence)
		POI_sequences.append(user_POI_sequences)
		
	reIndexed_POI_sequences, POI_reIndex_mapping = _reIndex_3d_list(np.array(POI_sequences,dtype=object))
	return reIndexed_POI_sequences, POI_reIndex_mapping



def generate_category_sequences(input_data, visit_sequence_dict):
	cate_sequences = []
	for user in visit_sequence_dict:
		user_cate_sequences = []
		for seq in visit_sequence_dict[user]:
			cate_sequence = []
			for visit in seq:
				if visit != -1:
					cate_sequence.append(input_data['L2_id'][visit])
				else: 
					cate_sequence.append(-1)
			user_cate_sequences.append(cate_sequence)
		cate_sequences.append(user_cate_sequences)
	reIndexed_cate_sequences, cate_reIndex_mapping = _reIndex_3d_list(np.array(cate_sequences,dtype=object))
	return reIndexed_cate_sequences, cate_reIndex_mapping

def generate_region_sequences(input_data, visit_sequence_dict):
	region_sequences = []
	for user in visit_sequence_dict:
		user_region_sequences = []
		for seq in visit_sequence_dict[user]:
			region_sequence = []
			for visit in seq:
				if visit != -1:
					region_sequence.append(input_data['RegionId'][visit])
				else: 
					region_sequence.append(-1)
			user_region_sequences.append(region_sequence)
		region_sequences.append(user_region_sequences)
	reIndexed_region_sequences, region_reIndex_mapping = _reIndex_3d_list(np.array(region_sequences,dtype=object))
	return reIndexed_region_sequences, region_reIndex_mapping


def generate_time_sequences(input_data, visit_sequence_dict):
	time_sequences = []
	for user in visit_sequence_dict:
		user_time_sequences = []
		for seq in visit_sequence_dict[user]:
			time_sequence = []
			for visit in seq:
				if visit != -1:
					time_sequence.append(input_data['hour'][visit])
				else: 
					time_sequence.append(-1)
			user_time_sequences.append(time_sequence)
		time_sequences.append(user_time_sequences)
	reIndexed_time_sequences, time_reIndex_mapping = _reIndex_3d_list(np.array(time_sequences,dtype=object))
	return reIndexed_time_sequences, time_reIndex_mapping

def get_distance(pos1, pos2):
	lat1, lon1 = pos1
	lat2, lon2 = pos2
	
	dlat = lat2 - lat1
	dlon = lon2 - lon1
	
	a = math.sin(math.radians(dlat / 2)) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(dlon / 2)) ** 2
	c = 2 * math.asin(math.sqrt(a))
	r = 6371 
	h_dist = c * r
	
	return h_dist

def generate_dist_sequences(input_data, visit_sequence_dict):
	dist_sequences = []
	max_dist = 0
	for user in visit_sequence_dict:
		user_dist_sequences = []	
		for seq in visit_sequence_dict[user]:		
			dist_sequence = []	
			for pos, visit in enumerate(seq):
				if pos == 0: 
					dist_sequence.append(0)		
				elif visit != -1:
					lat1 = input_data['Latitude'][visit]
					lon1 = input_data['Longitude'][visit]
					lat2 = input_data['Latitude'][seq[pos-1]]
					lon2 = input_data['Longitude'][seq[pos-1]]
					
					dist = get_distance((lat1,lon1), (lat2,lon2))
					dist_sequence.append(math.ceil(dist))
					max_dist = max(max_dist, math.ceil(dist))
				else: 
					dist_sequence.append(-1)
			user_dist_sequences.append(dist_sequence)
		dist_sequences.append(user_dist_sequences)
	return np.array(dist_sequences), max_dist

def generate_region_dist_sequences(input_data, visit_sequence_dict):
	dist_sequences = []
	max_dist = 0
	for user in visit_sequence_dict:
		user_dist_sequences = []	
		for seq in visit_sequence_dict[user]:		
			dist_sequence = []	
			for pos, visit in enumerate(seq):
				if pos == 0: 
					dist_sequence.append(0)		
				elif visit != -1:
					lat1 = input_data['R_Latitude'][visit]#
					lon1 = input_data['R_Longitude'][visit]
					lat2 = input_data['R_Latitude'][seq[pos-1]]
					lon2 = input_data['R_Longitude'][seq[pos-1]]
					dist = get_distance((lat1,lon1), (lat2,lon2))
					dist_sequence.append(math.ceil(dist))
					max_dist = max(max_dist, math.ceil(dist))
				else:
					dist_sequence.append(-1)
			user_dist_sequences.append(dist_sequence)
		dist_sequences.append(user_dist_sequences)
	return np.array(dist_sequences), max_dist


def _reIndex_3d_list(input_list):
	flat_list = _flatten_3d_list(input_list)
	index_map = np.unique(flat_list)
	if index_map[0] == -1:
		index_map = np.delete(index_map, 0)
	reIndexed_list = [] 
	for user in input_list:
		reIndexed_user_list = [] 
		for seq in user:
			reIndexed_user_list.append([_old_id_to_new(index_map, poi) if poi != -1 else -1 for poi in seq])	
		reIndexed_list.append(reIndexed_user_list)
	reIndexed_list = np.array(reIndexed_list,dtype=object)
	check_list = _flatten_3d_list(reIndexed_list)
	if -1 in check_list:
		check_list = check_list[ check_list >= 0 ]
	check_is_consecutive(check_list, 0) 
	return reIndexed_list, index_map

def _flatten_3d_list(input_list):
	twoD_lists = input_list.flatten()
	return np.hstack([np.hstack(twoD_list) for twoD_list in twoD_lists])

def _old_id_to_new(mapping, old_id):
	return np.where(mapping == old_id)[0].flat[0]

def _new_id_to_old(mapping, new_id):
	return mapping[new_id]
def check_is_consecutive(check_list, start_index):
	assert check_list.max() == len(np.unique(check_list)) + start_index - 1, 'ID is not consecutive'
def get_training_testing(sequence):
    all_training_samples=[]
    all_validation_samples=[]
    all_training_validation_samples=[]
    all_testing_samples=[]
    for user_samples in sequence:
        N = len(user_samples)
        if N > 9:
            train_test_boundary = int(0.8*N)
            vaild_test_boundary = int(0.5*len(user_samples[train_test_boundary:]))
            train_valid_boundary = int(0.9*N)
            all_training_samples.append(user_samples[:train_test_boundary])
            all_validation_samples.append(user_samples[train_test_boundary:][:vaild_test_boundary])
            all_training_validation_samples.append(user_samples[:train_valid_boundary])
            all_testing_samples.append(user_samples[train_test_boundary:][vaild_test_boundary:])
        else:
            all_testing_samples.append([user_samples[-1]])
            all_validation_samples.append([user_samples[-2]])
            if len(user_samples)<4:
                all_training_samples.append([user_samples[0]])
                all_training_validation_samples.append(user_samples[:-1])
            else:
                all_training_samples.append(user_samples[:-2])
                all_training_validation_samples.append(user_samples[:-1])
    return all_training_samples,all_validation_samples,all_testing_samples,all_training_validation_samples
def get_PreTarget(data):
    all_pre_list=[]
    all_traget_list=[]
    for i in data:
        user_target=[]
        user_pre=[]
        for j in i:
            user_target.append(j[-1])
            user_pre.append(j[:-1])
        all_pre_list.append(user_pre)
        all_traget_list.append(user_target)
    return all_pre_list,all_traget_list
def get_tr_va_te_data(data):
    print(len(data))
    cat_counter = collections.Counter(data['Category'])
    POI_counter = collections.Counter(data['VenueId'])
    cat_id_mapping = dict(zip(cat_counter.keys(), np.arange(len(cat_counter.keys()))))
    POI_id_mapping = dict(zip(POI_counter.keys(), np.arange(len(POI_counter.keys()))))
    data['L2_id'] = data['Category'].apply(lambda x: cat_id_mapping[x])
    data['Location_id'] = data['VenueId'].apply(lambda x: POI_id_mapping[x])
    data['hour'] = pd.to_datetime(data['Local_Time_True']).dt.hour
    visit_sequences, max_seq_len, valid_visits, user_reIndex_mapping = generate_sequence(data, min_seq_len=2, min_seq_num=3)
    assert bool(visit_sequences), 'no qualified sequence after filtering!' # check if output sequence is empty
    print(len(visit_sequences))
    visit_sequences, ground_truth_dict = aug_sequence(visit_sequences, min_len=3)
    POI_sequences, POI_reIndex_mapping =generate_POI_sequences(data, visit_sequences)
    cate_sequences, cate_reIndex_mapping = generate_category_sequences(data, visit_sequences)
    region_sequences, region_reIndex_mapping = generate_region_sequences(data, visit_sequences)
    time_sequences, time_reIndex_mapping = generate_time_sequences(data, visit_sequences)
    dist_sequences, max_dist = generate_dist_sequences(data, visit_sequences)
    region_dist_sequences, max_dist = generate_dist_sequences(data, visit_sequences)
    POI_training_samples,POI_validation_samples,POI_testing_samples,POI_training_validation_samples=get_training_testing(POI_sequences)
    region_training_samples,region_validation_samples,region_testing_samples,region_training_validation_samples=get_training_testing(region_sequences)
    category_training_samples,category_validation_samples,category_testing_samples,category_training_validation_samples=get_training_testing(cate_sequences)
    time_training_samples,time_validation_samples,time_testing_samples,time_training_validation_samples=get_training_testing(time_sequences)
    distance_training_samples,distance_validation_samples,distance_testing_samples,distance_training_validation_samples=get_training_testing(dist_sequences)
    regi_distance_training_samples,regi_distance_validation_samples,regi_distance_testing_samples,regi_distance_training_validation_samples=get_training_testing(region_dist_sequences)


    POI_train_pre,POI_train_target=get_PreTarget(POI_training_samples)
    POI_valid_pre,POI_valid_target=get_PreTarget(POI_validation_samples)
    POI_test_pre,POI_test_target=get_PreTarget(POI_testing_samples)
    POI_train_valid_pre,POI_train_valid_target=get_PreTarget(POI_training_validation_samples)

    region_train_pre,region_train_target=get_PreTarget(region_training_samples)
    region_valid_pre,region_valid_target=get_PreTarget(region_validation_samples)
    region_test_pre,region_test_target=get_PreTarget(region_testing_samples)
    region_train_valid_pre,region_train_valid_target=get_PreTarget(region_training_validation_samples)

    category_train_pre,category_train_target=get_PreTarget(category_training_samples)
    category_valid_pre,category_valid_target=get_PreTarget(category_validation_samples)
    category_test_pre,category_test_target=get_PreTarget(category_testing_samples)
    category_train_valid_pre,category_train_valid_target=get_PreTarget(category_training_validation_samples)

    time_train_pre,time_train_target=get_PreTarget(time_training_samples)
    time_valid_pre,time_valid_target=get_PreTarget(time_validation_samples)
    time_test_pre,time_test_target=get_PreTarget(time_testing_samples)
    time_train_valid_pre,time_train_valid_target=get_PreTarget(time_training_validation_samples)

    distance_train_pre,distance_train_target=get_PreTarget(distance_training_samples)
    distance_valid_pre,distance_valid_target=get_PreTarget(distance_validation_samples)
    distance_test_pre,distance_test_target=get_PreTarget(distance_testing_samples)
    distance_train_valid_pre,distance_train_valid_target=get_PreTarget(distance_training_validation_samples)

    regi_distance_train_pre,regi_distance_train_target=get_PreTarget(regi_distance_training_samples)
    regi_distance_valid_pre,regi_distance_valid_target=get_PreTarget(regi_distance_validation_samples)
    regi_distance_test_pre,regi_distance_test_target=get_PreTarget(regi_distance_testing_samples)
    regi_distance_train_valid_pre,regi_distance_train_valid_target=get_PreTarget(regi_distance_training_validation_samples)

    user_train=(POI_train_pre,POI_train_target,category_train_pre,category_train_target,region_train_pre,region_train_target,time_train_pre,time_train_target,distance_train_pre,distance_train_target,regi_distance_train_pre,regi_distance_train_target)
    user_valid=(POI_valid_pre,POI_valid_target,category_valid_pre,category_valid_target,region_valid_pre,region_valid_target,time_valid_pre,time_valid_target,distance_valid_pre,distance_valid_target,regi_distance_valid_pre,regi_distance_valid_target)
    user_train_valid=(POI_train_valid_pre,POI_train_valid_target,category_train_valid_pre,category_train_valid_target,region_train_valid_pre,region_train_valid_target,time_train_valid_pre,time_train_valid_target,distance_train_valid_pre,distance_train_valid_target,regi_distance_train_valid_pre,regi_distance_train_valid_target)
    user_test=(POI_test_pre,POI_test_target,category_test_pre,category_test_target,region_test_pre,region_test_target,time_test_pre,time_test_target,distance_test_pre,distance_test_target,regi_distance_test_pre,regi_distance_test_target)
        
    return user_train,user_valid,user_train_valid,user_test
def flat_list(l):
    li=[]
    for i in range(len(l)):
        for j in range (len(l[i])):
            li.append(l[i][j])
    return li
def region_seq(data):
    cat_counter = collections.Counter(data['Category'])
    POI_counter = collections.Counter(data['VenueId'])
    cat_id_mapping = dict(zip(cat_counter.keys(), np.arange(len(cat_counter.keys()))))
    POI_id_mapping = dict(zip(POI_counter.keys(), np.arange(len(POI_counter.keys()))))
    data['L2_id'] = data['Category'].apply(lambda x: cat_id_mapping[x])
    data['Location_id'] = data['VenueId'].apply(lambda x: POI_id_mapping[x])
    data['hour'] = pd.to_datetime(data['Local_Time_True']).dt.hour
    visit_sequences, max_seq_len, valid_visits, user_reIndex_mapping = generate_sequence(data, min_seq_len=2, min_seq_num=3)
    assert bool(visit_sequences), 'no qualified sequence after filtering!' 
    visit_sequences, ground_truth_dict = aug_sequence(visit_sequences, min_len=3)
    POI_sequences, POI_reIndex_mapping =generate_POI_sequences(data, visit_sequences)
    cate_sequences, cate_reIndex_mapping = generate_category_sequences(data, visit_sequences)
    region_sequences, region_reIndex_mapping = generate_region_sequences(data, visit_sequences)
    time_sequences, time_reIndex_mapping = generate_time_sequences(data, visit_sequences)
    POI_dist_sequences, max_dist = generate_dist_sequences(data, visit_sequences)
    region_dist_sequences, max_dist = generate_region_dist_sequences(data, visit_sequences)
    return POI_sequences,cate_sequences,region_sequences,time_sequences,POI_dist_sequences,region_dist_sequences
def get_In_Cross_region_seq(region_sequences):
    group_InRegion_index=[]
    group_CrossRegion_index=[]
    for x in region_sequences:
        InRegion_index=[]
        CrossRegion_index=[]
        for i in range(len(x)):
            if len(set(x[i]))==1:
                InRegion_index.append(i)
            else:
                CrossRegion_index.append(i)
        group_InRegion_index.append(InRegion_index)
        group_CrossRegion_index.append(CrossRegion_index)
    return group_InRegion_index,group_CrossRegion_index

def remove_zero_element(data):
    no_zero=[]
    for i in range(len(data)):
        no_zero_data=[]
        # print('i',i)
        if i%2==0:
            for seq in data[i]:
                b= [seq_item + 1 for seq_item in seq]
                no_zero_data.append(b)
        else:
            no_zero_data= [j + 1 for j in data[i]]
        no_zero.append(no_zero_data)
    return(no_zero)
def flat_list(l):
    li=[]
    for i in range(len(l)):
        #print(l[i])
        for j in range (len(l[i])):
            li.append(l[i][j])
    return li
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx 
def get_adj_matrix_InDegree(data,max_item):
    max_node=max_item
    adj_matrix = np.zeros((max_node, max_node))#
    for seq in data:
        for i in range(len(seq)-1):
            adj_matrix[seq[i+1]][seq[i]]=1
    adj_matrix = sp.coo_matrix(adj_matrix, dtype=np.int16)
    adj_matrix=normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
    return adj_matrix.toarray()

def get_adj_matrix_InDegree_txt(train_data,train_valid_data,test_data):
    all_i=flat_list(train_valid_data[0])+train_valid_data[1]+flat_list(test_data[0])+test_data[1]
    POI_n_node=max(np.unique(flat_list(all_i)))+2
    train_data= [flat_list(le) for le in train_data]
    train_valid_data= [flat_list(le) for le in train_valid_data]
    train_data=remove_zero_element(train_data)
    train_valid_data=remove_zero_element(train_valid_data)
    POI_adj_train_valid=get_adj_matrix_InDegree(train_valid_data[0],POI_n_node)
    POI_adj_train=get_adj_matrix_InDegree(train_data[0],POI_n_node)
    return POI_adj_train_valid,POI_adj_train
def get_GroupData(data_sanmeR_index,data):
	group_data=defaultdict(list)
	for i in range(len(data_sanmeR_index)):
		if len(data_sanmeR_index[i])==0:
			continue
		else:
			for j in data_sanmeR_index[i]:
				group_data[1].append(data[i][j])
	return group_data



def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max
def remove_zero_element(data):
    no_zero=[]
    for i in range(len(data)):
        no_zero_data=[]
        if i%2==0:
            for seq in data[i]:
                b= [seq_item + 1 for seq_item in seq]
                no_zero_data.append(b)
        else:
            no_zero_data= [j + 1 for j in data[i]]
        no_zero.append(no_zero_data)
    return(no_zero)


def flat_list(l):
    li=[]
    for i in range(len(l)):
        #print(l[i])
        for j in range (len(l[i])):
            li.append(l[i][j])
    return li

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max
class Data():
    def __init__(self, data, shuffle=False):

        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices
    
    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        return mask, targets,inputs



class Data_GroupLabel():
    def __init__(self, data, shuffle=False):

        inputs = data
        self.inputs = np.asarray(inputs)
        self.length = len(inputs)
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices
    
    def get_slice(self, i):
        inputs= self.inputs[i]
        return inputs
