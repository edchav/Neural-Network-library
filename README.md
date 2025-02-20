# Neural-Network-library

This repository contains custom implementation of a neural network library

# Requirements
1. Python
2. Numpy
3. Pandas
4. Scikit-learn
5. Matplotlib

# Run the program
1. Pretrained models are in the 'Models' folder, I suggest opening the notebooks for 'predict_trip_duration_{i} and running blocks underneath headings 'Load Model {i} Weights'

    - Since the NYC dataset is very large training may take some time even with early stopping
    - Validation checks should be checked every epoch not every step this casued a lot of overhead due to the flucuation of batch loss

# Folders and Files
The 'Data' folder contains the NYC dataset. The 'Models' folder contains the pretrained models for 'xor_problem', 'predict_trip_duration_{i}'. The 'Plots' folder contains the loss graphs. 