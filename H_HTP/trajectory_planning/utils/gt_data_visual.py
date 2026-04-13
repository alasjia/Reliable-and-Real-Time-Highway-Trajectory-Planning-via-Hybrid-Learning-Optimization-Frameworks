import os
import matplotlib.pyplot as plt
from tra_plannning.data4VN import get_features_we_want, get_frequency_we_want

def ev_gt_visualization(track_df, rec_id, case_id,  save_dir1):
    features_we_want = get_features_we_want(track_df)
    frequency_we_want = get_frequency_we_want(features_we_want)
    
    track_array = frequency_we_want.to_numpy()      #(the step number, 54 = 6*9)
    steps = [i for i in range(track_array.shape[0]) ]

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    # Plot the longitudinal planning results in the first row
    axs[0, 0].plot(steps, track_array[:, 0], label='Longitudial displacement')
    axs[0, 1].plot(steps, track_array[:, 2], label='Longitudial speed')
    axs[0, 2].plot(steps, track_array[:, 4], label='Longitudial acceleration')

    # Plot the lateral planning results in the second row
    axs[1, 0].plot(steps, track_array[:, 1], label='Lateral displacement')
    axs[1, 1].plot(steps, track_array[:, 3], label='Lateral speed')
    axs[1, 2].plot(steps, track_array[:, 5], label='Lateral acceleration')


    # Set the title and labels for both subplots
    for i in range(2):
        for j in range(3):
            axs[i, j].set_xlabel('Step')
            
    for i in range(2):
        for j in range(3):
                    axs[i, j].set_ylabel(f'{["Displacement (m)","Speed (m/s)","Acceleration (m/s^2)"][j]}') 
                    
    # for i in range(2):
    #         axs[i, 0].set_title(f'{["Longitudinal","Lateral"][i]} Planning Results')

    # Add a legend for all subplots
    for i in range(2):
        for j in range(3):
            axs[i, j].legend()

    # Set the title for the figure
    fig.suptitle(f'Case {case_id}: Ground Truth Kinematics Results')

    # Display the plot
    plt.show()
    
    # # Save the figure
    # file_name = f'rec{rec_id+1}_case{case_id+1}'  # index starts from 1
    # plt.savefig(os.path.join(save_dir1, file_name+'.jpg'))
    
    return 1