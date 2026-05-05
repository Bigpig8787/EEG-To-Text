@echo off
REM Convert all 5 ZuCo .mat datasets to .pickle (rawData included).
REM v1 subjects: Z* (12 subjects)  ;  v2 subjects: Y* (18 subjects)
echo Converting ZuCo .mat files to .pickle with rawData...
python util/construct_dataset_mat_to_pickle_v1.py -t task1-SR
python util/construct_dataset_mat_to_pickle_v1.py -t task2-NR
python util/construct_dataset_mat_to_pickle_v1.py -t task3-TSR
python util/construct_dataset_mat_to_pickle_v2.py -t task2-NR-2.0
python util/construct_dataset_mat_to_pickle_v2.py -t task2-TSR-2.0
echo Done!
pause
