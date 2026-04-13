Description of this section:
1. Process all data—including non-lane-change cases—into a standard format by removing the lane-change filtering step from the original processing.
2. Categorize SVs into 8 groups (according to the original data classification). For each trajectory, add SV information in frame order and store it in the dataset for training.
3. Extract the required features for the dataset and convert them into a format suitable for input into the neural network model (evdata_pre, svdata_pre).


if you are not interested in the data processing, you can ignore this section and use the dataset in the "DataSets/dataset_after_processing" directly.



if you are not interested in the process of converting data in "dataset_after_dp" to ".pt" files, you can ignore the usage of [HighD_datapre.py](HighD_datapre.py) and directly use "DataSets/dataset_after_processing/PyG_DataSet"
