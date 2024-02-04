Code accompanying the paper "Reproducibility study of "LICO: Explainable Models with Language-Image Consistency"".

To install the necessary packages, run
``conda env create -f environment.yml``.

To run the code, activate the environment, and then run
``python train.py --save_model_name [NAME] --train_dataset [cifar10/cifar100] --data_root [dir containing cifar10, cifar100, imagenette folders]``. For the rest, the default parameters are all the same as in the paper.

To evaluate the model, #TODO
