## About the vic3 project

For technical information scroll down

### Introduction

There's grand strategy game publisher known as Paradox, with historical grand strategy games of different franchises and different historical time periods. Each of their titles only focus on a single time period, but the different time periods span from the middle age to the 2nd world war.

Since the games are historical, this naturally leads one to wonder, what if one can inherit a world from the middle age in one game, and continue it on the next franchise through the age of discovery, of reformation, and then onto the next in the Victoria era, and so on?

There's a talented group of programmers creating and maintaining such 'converters'. However, since the titles are different games in different time periods, they focus on different mechanics, and the 'world information' of one game is not necessarily present in the next, and vice versa.
These dissimilarities and information gap need to be bridged somehow.

This project is intended to assist with bridging the information gap between the world of 'Europa Universalis 4' and 'Victoria 3'.
In particular, the world of EU4 is unique in its absence of population data, yet population is the core mechanics of Victoria 3.

### Motivation

Along with population, the world of Victoria 3, from now on vic3, is 'populated' with resources, terrains, and buildings, among other things.
In the real world, resources, food in particular, is the main determinant of livelihoods and population density.
The same could be said of geographic terrain, its biomes, and existing structures are potential indicators of civilization as well.
I believe a relation should exist between physical geography and population of a 'state', the basic geographic unit of vic3.
A ML model could potentially learn this base pattern of relationships, and from the base pattern a conjecture bridging EU4 save data with the plausible population of an alternate world could be formed.
The main goal would be such that an alternate history where London is a backwater of a minuscule principality should not have the same level of population as historical London in vic3 timeline, yet it should be within physical plausibility.

### Components

The project is not 100% modular and command-line-compatible yet, but these are its components.

 - **data_preparation.py** : This is the data preparation modules that load vic3 game files and compile them into various data jsons. All functions have 'redo' parameter is set to automatic False, however, other modules that call them don't specify 'redo' so if you ever need to update data files, you'll need to delete them to force a redo.
 - **ml.py** : Contains 2 main functions that create an ML model, one of which returns it, while the other returns 'evaluation metrics' and is used for the purpose of testing feature combinations or making prediction map. Bundled with it are the various data transformation functions, many one-hots for different feature sets, and many more feature engineering modules will be added.
 - **test_features.py** : Spawns 12 threads to test including/excluding feature combinations via leave-one-out cross validation. This can take very long to run depending number of features being tested, as it brute-forces its way through all the possible combinations.
 - **get_prediction_maps.py** : Using a specified set of features, using leave-one-out cross validation, spawn up to 12 threads to overlay onto province map the color code of where predictions are 'correct' (grey), 'should be higher' (white), or 'should be lower' (black). 'Correct' are defined with different thresholds.
 - **get_model.py** : Using a specified set of features, train model on all data points and yield the resulting coefficients with feature names in json format, as well as a sample data frame of what the model input should look like.

### Technical Information

This will be divided roughly into user guide and tech stack information

#### User Guide

Normal users should only ever need to touch 3 entry files, **with the exception of 1st time usage**, where you might need to edit vic3 data directory path to be where you installed it.

 - **test_features.py** : Simply edit the *all_possible* variable to contain feature sets you want to test. Feature names much match the given commented out line. Log and square feature are not compatible to all sets yet, this was added last minute. You should not have to use log and square feature anyway. I've tried and can tell none of these are useful.
 - **get_prediction_maps.py** : Simply edit the *sets_to_regress* to test the metrics of the model using desired features.
 - **get_model.py** : Simply edit the 2nd argument as you'd edit *sets_to_regress* to get the desired equation in json format. The 1st argument might be removed in future updates.

#### Tech Stack

The ML aspect of the project is admittedly not much to write home about, utilizing only Linear Regression. This is because the converter team codes with C++, I want something simple that can be simply recreated by them via a single equation. Using a decision tree I'll have to have them translate a set of rules into a decision pipeline, which is way more complicated even if using a single tree (no Gradient Boosting, Random Forest, etc.). Neural Networks would be equally as complicated as trees to transfer across platform. Granted, I don't code on C++ and the team doesn't do Data Science, so a single linear equation is desirable.

In general you'll notice I use a lot of map-filter-reduce, because functional programming is my preferred method, even if Python's map isn't really parallel, and I don't adhere to it all the time and don't necessarily start developing with it.

 - **data_preparation.py** : is mainly about regex-ing text files to generate json format data. An irregularity is reading/transforming province map with cv2 and generating coast adjacency data.
 - **ml.py** : contains the ML code as well as many 'one-hot' transformers (which are actually just transforming dictionary data into new columns).
 - **test_features.py** : uses multiprocessing.Pool to map the feature testing function in parallel.
 - **get_prediction_maps.py** : contains usage of numpy.apply_over_axis to transform image data, along with usage of Python's multiprocessing and functional tools.
 - **get_model.py** : This is really just a wrapper to call a function in *ml.py*, just so that *ml.py* doesn't have to be touched or editted ever in normal usage.