# MCMG

This repository is the implementation of MCMG.

MCMG is short for 'A **M**ulti-**C**hannel Next POI Recommendation Framework with **M**ulti-**G**ranularity Check-in Signals'. It is a novel framework, equipped with three modules (i.e., global user behavior encoder, local multi-channel encoder, and region-aware weighting strategy), which is designed to achieve the next POI prediction by fusing multi-granularity check-in signals from two orthogonal perspectives, i.e., fine-coarse grained check-ins at either POI/region level or local/global level.


### File Descriptions

- `dataset/`
  - `CAL/`
    - `AdjacentMatrix_CAL.txt`: the adjacent matrix of Calgary;
    - `CAL_checkin.csv`: check-ins information of Calgary;
    - `test_CAL.txt`: test set of Calgary;
    - `test_group_label_CAL.txt`: group lables of trajectories in test set;
    - `train_CAL.txt`: training set of Calgary;
    - `train_group_label_CAL.txt`: group lables of trajectories in training set;
  - `NY/`
    - `NY.rar`: all the data files of New York;
  - `PHO/`
    - `PHO.rar`: all the data files of Phoenix;
  - `SIN/`
    - `SIN.rar`: all the data files of Singapore;
- `appendix.pdf`: the best parameter settings for all methods;
- `main.py`: main file;
- `model.py`: MCMG model file;
- `parameter_setting.py`: parameter settings file;
- `utils.py`: utils file.


### More Experimental Settings
- Environment
  - Our proposed MCMG is implemented using pytorch 1.7.1, with Python 3.6.3 from Anaconda 4.3.30. All the experiments are carried out on a machine with Windows 10, Intel CORE i7-8565U CPU, NIVIDA GeForce RTX 2080 and 16G RAM. The following packages are needed (along with their dependencies):
    - cuda==11.0
    - hyperopt==0.2.4
    - numpy==1.19.5
    - pandas==1.1.5
    - progressbar==2.5
    - python==3.6.3
    - scipy==1.5.4
    - torch==1.7.1
- Data Preprocessing
  - Following state-of-the-arts, for each user, we chronologically divide his check-in records into different trajectories by day, and then take the earlier 80% of his trajectories as training set; the latest 10% of trajectories as the test set; and the rest 10% as the validation set. Besides, we filter out users with less than three trajectories.


### How To Run
```
$ python main.py (note: use -h to check optional arguments)
```
