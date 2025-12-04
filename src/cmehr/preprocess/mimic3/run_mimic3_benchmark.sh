# 1. Extract subjects from MIMIC-III CSVs
python -m mimic3benchmark.scripts.extract_subjects /data/mimiciii/physionet.org/files/mimiciii/1.4 data/root/
# 2. Extract events from MIMIC-III CSVs
python -m mimic3benchmark.scripts.validate_events data/root/
# 3. Extract episodes from subjects
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
# 4. Split episodes into train and test sets
python -m mimic3benchmark.scripts.split_train_and_test data/root/
# 5. Create in-hospital mortality and phenotyping datasets
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/
# 6. Split the dataset into training and validation sets
python -m mimic3models.split_train_val data/root/
# for each task 
python -m mimic3models.split_train_val data/root/ data/in-hospital-mortality/ --valset mimic3models/resources/valset_in-hospital-mortality.csv
python -m mimic3models.split_train_val data/root/ data/decompensation/ --valset mimic3models/resources/valset_decompensation.csv
python -m mimic3models.split_train_val data/root/ data/length-of-stay/ --valset mimic3models/resources/valset_length-of-stay.csv
python -m mimic3models.split_train_val data/root/ data/phenotyping/ --valset mimic3models/resources/valset_phenotyping.csv
python -m mimic3models.split_train_val data/root/ data/phenotyping/ --valset mimic3models/resources/valset_multitask.csv
# 7. Create a benchmark dataset
python -m mimic3models.create_iiregular_ts --task ihm
python -m mimic3models.create_iiregular_ts --task decompensation
python -m mimic3models.create_iiregular_ts --task length-of-stay
python -m mimic3models.create_iiregular_ts --task phenotyping
python -m mimic3models.create_iiregular_ts --task multitask
