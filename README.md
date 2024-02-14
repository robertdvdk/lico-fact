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
``python train.py --train_dataset imagenette_160 --save_model_name [NAME] --data_root [dir containing imagenette_160 folder] --width 8 --batch_size 32 --image_feature_dim 1600``

## Evaluating the model

### Test accuracy
To evaluate the model's test accuracy, activate the environment and then use the following commands:

CIFAR10:
``python evaluate.py --test_dataset cifar10 --model_path [path/to/model] --data_root [dir containing cifar10 folder]``

CIFAR100:
``python evaluate.py --test_dataset cifar100 --model_path [path/to/model] --data_root [dir containing cifar100 folder] --width 8``

Imagenette:
``python evaluate.py --test_dataset imagenette_160 --model_path [path/to/model] --data_root [dir containing imagenette folder] --width 8 --image_feature_dim 1600``

### Insertion-deletion score
To obtain the insertion deletion scores, activate the environment and then use the following commands:

CIFAR10:
``python insertion_deletion.py --test_dataset cifar10 --model_path [path/to/model] --data_root [dir containing cifar10 folder] --saliency_method [saliency method to use]``

CIFAR100:
``python evaluate.py --test_dataset cifar100 --model_path [path/to/model] --data_root [dir containing cifar100 folder] --saliency_method [saliency method to use] --width 8``

Imagenette:
``python evaluate.py --test_dataset imagenette_160 --model_path [path/to/model] --data_root [dir containing imagenette folder] --saliency_method [saliency method to use] --width 8 --image_feature_dim 1600 --pixel_batch_size 922``

