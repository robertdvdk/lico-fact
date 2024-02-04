Code accompanying the paper "Reproducibility study of "LICO: Explainable Models with Language-Image Consistency"".

## Setting up the environment
To install the necessary packages, run
``conda env create -f environment.yml``.

## Training the model
To run the model as in our experiments, activate the environment and then use the following commands:

CIFAR10:
``python train.py --train_dataset cifar10 --save_model_name [NAME] --data_root [dir containing cifar10 folder]``

CIFAR100:
``python train.py --train_dataset cifar100 --save_model_name [NAME] --data_root [dir containing cifar100 folder] --width 8``

Imagenette:
``python train.py --train_dataset imagenette_160 --save_model_name [NAME] --data_root [dir containing imagenette_160 folder] --width 8 --batch_size 32``

## Evaluating the model
To evaluate the model, #TODO
