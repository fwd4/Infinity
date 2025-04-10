import pickle

# with open(f'outputs/codes/test_pixel_partialblock_data_3d_school.pkl', 'rb') as file:
#     data = pickle.load(file)

# for si, value in data.items():
#     for i in range(len(value)):
#         last_stage = value[i]

import glob
file_paths = glob.glob('/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/codes/test_pixel_partialblock_prob_*.pkl')
print(len(file_paths))