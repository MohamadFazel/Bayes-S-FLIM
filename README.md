# Spectral-FLIM
Spectral-FLIM is a Python-based Bayesian tool designed to handle the intricacies of Spectral Fluorescence Lifetime Imaging Microscopy (FLIM), providing advanced capabilities for spectral analysis in fluorescence microscopy.
<!-- ## Features -->

## Installation
### 1. Verify GPU and Drivers
For NVIDIA GPUs: Install drivers from NVIDIA and verify with:
```
nvidia-smi
```
### 2. Install Miniconda
Download and install Miniconda[https://docs.anaconda.com/miniconda/]
### 3. Create and Activate Conda Environment
```
conda create -n your_env_name python=3.9
conda activate your_env_name
```
### 4. Install Dependencies
Ensure you are in the project directory, then run:
```
pip install -r requirements.txt
```
### 5.Verify GPU Integration with CuPy
You can test if CuPy recognizes your GPU by running a simple random number generation test:
```
import cupy as cp

# Generate random numbers on GPU
random_numbers = cp.random.rand(5)
print(random_numbers)
```
## Usage
To use this application, follow these simple steps:
### 1. Launch the Application:
Open your terminal, navigate to the project directory, and run:
```
python app.py
```

### 2. Choose Input Data and set params:
* Once the GUI is open, you will see to select your input data file.
* Ensure the input data is in .mat format and includes two keys:
    - Dt
    - Lambda
* Set other parameters in the GUI.
* For more setting you have to open Setting from Menu Bar and choose hyperparameter settings.
###  3. Process and Analyze
* Choose your input file and set the parameters. Once ready, click RUN
* Monitor the progress through the progress bar displayed.
* Upon completion, you have the option to plot or export the results.
## Contributing
Dr Mohamadreza Fazel, Reza Hoseini, Dr Steve Presse
## License
Presse Lab