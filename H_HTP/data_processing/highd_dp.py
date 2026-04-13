#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu June 6 2024

@author: yujia lu

Compared with the old version of highd_dp:
This code changes the SV (surrounding vehicle) search method back to searching for a specific vehicle for each SV position.
2024.7.2: It was found that if SVs are searched based on the most frequent occurrence, two positions may correspond to the same vehicle. Therefore, the search method is simplified again: the SVs in a certain frame are considered as the surrounding vehicles for that trajectory.
                   The problem with this approach is that for each position (front, rear, front-left, etc.), only one vehicle is searched/fixed at that position for the entire trajectory. In reality, during the progression of the trajectory:
                   1) The front SV may become the front-left SV, or the front-left may become the left-middle, causing confusion in the data features for each position in the observation, which increases the difficulty for the agent to understand.
                   2) There may be two (or more) SVs at the same position, and ignoring some SVs increases the loss of environmental information.
                   Issue 1) May be alleviated with the help of graph neural networks.
                   Issue 2) For now, since the duration of a single trajectory in the highD dataset is not very long (8 sec ~ 15 sec), the possibility of losing key SVs due to this simplification is not high.

'''


import pandas as pd
import numpy as np
import os, re
import matplotlib.pyplot as plt
import time
import sys
from torch.utils.data import Dataset, DataLoader


# Coordinate transformation due to vehicle driving direction; coordinate transformation to vehicle center based on vehicle dimensions
def get_same_direction(tracksMeta, tracks, transfer_y = 30):
    # Coordinate transformation due to vehicle driving direction: change the data of the upper lane (right to left) to be unified as left to right
    tracksMeta_upper = tracksMeta[tracksMeta['drivingDirection'] == 1]
    tracksMeta_upper_ids = tracksMeta_upper['id'].tolist()  # All upper lane trajectory IDs
    tracksMeta_lower = tracksMeta[tracksMeta['drivingDirection'] == 2]
    tracksMeta_lower_ids = tracksMeta_lower['id'].tolist()  # All lower lane trajectory IDs
    # Divide the trajectories by frame into two sets: upper_set and lower_set
    upper_set = tracks[tracks['id'].isin(tracksMeta_upper_ids)]
    lower_set = tracks[tracks['id'].isin(tracksMeta_lower_ids)]
    # The coordinate system records the top-left corner of the vehicle; convert to vehicle center based on vehicle dimensions
    # The trajectories of vehicles in the upper lane also need further direction conversion to unify as left to right
    x_upper = 400 - (upper_set['x'] + upper_set['width']*0.5)
    y_upper = transfer_y - (upper_set['y'] + upper_set['height']*0.5)
    x_lower = lower_set['x'] + lower_set['width']*0.5
    y_lower = lower_set['y'] + lower_set['height']*0.5

    # Create new dataframes
    new_upper_set = upper_set.copy(deep=True)  # Deep copy to avoid two dataframes sharing the same address
    new_lower_set = lower_set.copy(deep=True)
    # Assign values according to the processing logic
    new_upper_set['x'] = x_upper
    new_upper_set['y'] = y_upper
    new_upper_set['xVelocity'] = upper_set['xVelocity'] * (-1)
    new_upper_set['yVelocity'] = upper_set['yVelocity'] * (-1)
    new_upper_set['xAcceleration'] = upper_set['xAcceleration'] * (-1)
    new_upper_set['yAcceleration'] = upper_set['yAcceleration'] * (-1)

    new_lower_set['x'] = x_lower
    new_lower_set['y'] = y_lower
    
    new_tracks = pd.concat([new_upper_set, new_lower_set])
    return new_tracks

def get_transfer_y(recordingMeta):
    up_markings = recordingMeta['upperLaneMarkings']
    lo_markings = recordingMeta['lowerLaneMarkings']
    
    # Use str.split() to split by semicolon into a list  
    split_lists_up = up_markings.str.split(';')  
    # Convert each string to float  
    float_lists_up = split_lists_up.apply(lambda x: [float(i) for i in x]).iloc[0]  
    
    # Use str.split() to split by semicolon into a list  
    split_lists_lo = lo_markings.str.split(';')  
    # Convert each string to float  
    float_lists_lo = split_lists_lo.apply(lambda x: [float(i) for i in x]).iloc[0]  
    
    transfer_y = float_lists_lo[-1] + float_lists_up[0]   
    return transfer_y

def add_svdata_data(single_track, tracks): 
    '''
    Index(['frame', 'id', 'x', 'y', 'width', 'height', 'xVelocity', 'yVelocity',
       'xAcceleration', 'yAcceleration', 'frontSightDistance',
       'backSightDistance', 'dhw', 'thw', 'ttc', 'precedingXVelocity',
       'precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId',
       'leftFollowingId', 'rightPrecedingId', 'rightAlongsideId',
       'rightFollowingId', 'laneId'],
      dtype='object')
    '''
    # # For each SV position, determine the unique vehicle id based on the most frequent occurrence
    # sv_ids = []
    # for idx, sv_str in  enumerate(['precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId', 'leftFollowingId', 'rightPrecedingId', 'rightAlongsideId', 'rightFollowingId' ] ):
    #     sv_id_candi = single_track[sv_str].value_counts().index
    #     # Get the unique vehicle id
    #     if sv_id_candi[0] != 0:
    #         sv_id = sv_id_candi[0]
    #     elif sv_id_candi.size > 1:
    #         sv_id = sv_id_candi[1]
    #     else:
    #         sv_id = -1
    #     sv_ids.append(sv_id)
    
    # For each SV position, use the SV situation at the frame in the middle of the trajectory
    sv_ids = []
    for idx, sv_str in  enumerate(['precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId', 'leftFollowingId', 'rightPrecedingId', 'rightAlongsideId', 'rightFollowingId' ] ):
        sv_id = single_track.iloc[single_track.shape[0]//2][sv_str]
        sv_id = -1 if sv_id == 0 else sv_id # If there is no SV at this position, assign -1
        sv_ids.append(sv_id)
        
        # According to sv_ids, search for the corresponding rows in the original tracks data, align by 'frame', and add to single_track
        # Rename the new columns
        new_column_names =  ['frame', sv_str[:-5] + '_laneId', sv_str[:-5] + '_x', sv_str[:-5] + '_y', sv_str[:-5] + '_vx', sv_str[:-5] + '_vy', sv_str[:-5] + '_ax', sv_str[:-5] + '_ay']
        # if sv_id < 0:
        #     target_sv_track = pd.DataFrame(index=single_track.index, columns=new_column_names)  
        # else:
        target_sv_track = tracks[tracks['id'] == sv_id][ ['frame',  'laneId', 'x', 'y', 'xVelocity',  'yVelocity',  'xAcceleration', 'yAcceleration'  ] ].copy(deep=True) 
        target_sv_track.columns = new_column_names # ['preced_x', 'preced_y', 'preced_laneId']   # If target_sv_track is empty, merge can still proceed
        # Merge the data with the new DataFrame based on the 'frame' column
        single_track = pd.merge(single_track, target_sv_track, on='frame', how='left')
        
    return single_track   # DataFrames are not passed by reference!

''''
     frame  id        x      y  ...  laneId  leftFollow_x  leftFollow_y  leftFollow_laneId
0        1   6   66.695  20.31  ...       2        14.715         16.41                  3
1        2   6   67.595  20.30  ...       2        15.725         16.40                  3
2        3   6   68.535  20.30  ...       2        16.795         16.39                  3
3        4   6   69.485  20.30  ...       2        17.885         16.39                  3
4        5   6   70.455  20.29  ...       2        18.995         16.38                  3
..     ...  ..      ...    ...  ...     ...           ...           ...                ...
331    332   6  385.805  20.68  ...       2       370.375         15.97                  3
332    333   6  386.745  20.68  ...       2       371.405         15.97                  3
333    334   6  387.695  20.67  ...       2       372.435         15.98                  3
334    335   6  388.635  20.67  ...       2       373.475         15.98                  3
335    336   6  389.575  20.66  ...       2       374.505         15.98                  3

[336 rows x 28 columns]
'''

# Filter and obtain data with frames num > 200
def get_long_tracks(tracksMeta, tracks, length_min = 200):
    new_tracksMeta = tracksMeta[tracksMeta['numFrames'] > length_min]
    # Get all track ids with frame num > 200
    ids = new_tracksMeta['id'].tolist()  

    # Get data with frames num > 200
    long_tracks = tracks[tracks['id'].isin(ids)]
    
    
    # # Use list comprehension to get the first 200 frames of each group  
    # fixed_tracks_list = [group.iloc[:200] for _, group in long_tracks.groupby('id')]  
      
    # # Use pd.concat to merge all groups  
    # fixed_tracks = pd.concat(fixed_tracks_list, ignore_index=True)  
      
    return long_tracks

def execute_code(record_num):
    # Start timing
    tic = time.time()   
    
    # Use listdir() to traverse all highd data files in the target path
    read_dir = '/home/chwei/reliable_and_realtime_highway_trajectory_planning/data_processing/raw_highd_data'   #please replace with your own path
    files_ = os.listdir(read_dir)  # 60*4
    # Use regex to match files starting with '.~lock.' and ending with '#', filter out temporary files
    pattern = r'^\.~lock.*#$'  
    files = [file for file in files_ if not re.match(pattern, file)]
    files.sort()  
    
    
    # Storage path directory
    save_dir_csv = '/home/chwei/reliable_and_realtime_highway_trajectory_planning/data_processing/dataset_after_dp'   #please replace with your own path
    
    all_records_data = []
    
    for record in range(record_num):    #Customize the range of recordings to be processed
        # Read data 
        recordingMeta = pd.read_csv(os.path.join(read_dir, str(files[record*4 + 1])))
        tracks = pd.read_csv(os.path.join(read_dir, str(files[record*4 + 2 ])))
        tracksMeta = pd.read_csv(os.path.join(read_dir, str(files[record*4 + 3 ])))
        
        # Road width
        transfer_y = get_transfer_y(recordingMeta)

        # Standard format conversion + fixed length frames
        tracks_v1 = get_same_direction(tracksMeta, tracks, transfer_y)
        tracks_v2 = get_long_tracks(tracksMeta, tracks_v1)
        
        # Create a list to collect all DataFrames  
        single_tracks_list = []  
        # Extract all trajectory IDs from the dataframe
        tracks_ids = list(set(tracks_v2['id'].tolist()))
        tracks_ids.sort()  # sort() has no return value
        for id in tracks_ids:
            single_track = tracks_v2[tracks_v2['id'] == id].reset_index(drop=True).copy(deep=True)  # copy() to avoid affecting the original dataframe, reset_index(drop=True) to regenerate index
            single_track_ = add_svdata_data(single_track, tracks_v1)  # Surrounding vehicle information is in v1
            single_tracks_list.append(single_track_)  
            
        tracks_v3 = pd.concat(single_tracks_list, ignore_index=True)
        # Write the dataframe by recording
        tracks_v3.to_csv(os.path.join(save_dir_csv, 'hd_dataset_after_dp_'+str(record + 1)+'.csv'))
        
    # all_records_data.append(tracks_v3)
    
    # End timing
    toc = time.time()
    print('the total time is: '+ '%.2f' % (toc - tic) + ' seconds.')
    
    return 1

if __name__ == '__main__':
    record_n = 20    #Customize the range of recordings to be processed
    execute_code(record_n)



