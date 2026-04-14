@echo off
echo Converting ZuCo .mat files to .pickle with rawData...
python util/construct_dataset_mat_to_pickle_v1.py -t task1-SR
python util/construct_dataset_mat_to_pickle_v1.py -t task2-NR
python util/construct_dataset_mat_to_pickle_v1.py -t task3-TSR
python util/construct_dataset_mat_to_pickle_v2.py
echo Done!
pause
