# cs6910_assignment1

1. Install the required libraries
2. The repository contains the following:
- network.ipynb notebook where I have coded the Questions 1,2,3,4,5,6,7,8 and it also contains the output that I have got.
- sweep.ipnyb notebook which contains the sweep performed along with the output( Questions 4,5,6). The plots are available in wandb report.
- the run command notebook which I have used to take arguments and train and test using command line arguments.( and the final one was run on default arguments which gave best accuracy)
- the mnist runcommand basically contains the run commands which was used for Question 10 to find out accuracy on MNIST dataset.
- train.py which can be used to train and test the model.

3. Instructions for training and testing the model:
- the repository contains the python file "train.py" 
- it can be used to train and test on the required parameters.
- Please note that I have given my project name (i.e. assignment1 ) and my entity name(i.e. cs22m025) as default arguments for the wandb project name and wandb entity name.
- I have hardcoded my key . So if it is needed to run the train.py on different projectname and different wandb then please change the key.
- It can be run on a command prompt by providing the required arguments.
- The same can also be run using the runcommand notebook by uploading the train.py as dataset(functionality in kaggle) and then providing the required path.

4. If a new sweep is to be run, then sweep agent call in sweep.ipnyb can be called with the required count. This can be done by modifying the count in sweep agent call present in sweep.ipnyb
