# Pix2Pix Tensorflow

This is a general purpose implementation of the pix2pix algorithm for image to image translation. 
This algorithm is based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

Code in this repo has been heavily borrowed from [this](https://github.com/affinelayer/pix2pix-tensorflow) implementation
of the pix2pix tensorflow. But the linked code is not easy to use as I have experienced it first hand while 
working on a related project.

This repo is an attempt to make the pix2pix implementation easily approachable for training and testing. The provided code 
also trains faster by removing some of the unnecessary operations. As such, there is some difference between how to define 
input and output in the linked repo and this one.

A guide on how to use this code for image to image translation is provided.

---

### Dependencies

You should have the following dependencies installed in your system to run this code.

- `python 3` 
- `tensorflow==1.10.0`

### Getting Started

#### 1. Training

The goal in image to image translation is to convert the input image A to target image B.

![A to B](https://i.ibb.co/pJQSsL3/ab.png)

Example:

![Example](https://i.ibb.co/bb84FZw/image-3.jpg)

You can also specify the mapping `AtoB` or `BtoA` in `config.py`.

For training, you should generate such images and put all the training images in the `train_data` folder in the root directory.

Alternatively, you can also use `--input-dir` flag to set your custom input directory.

e.g

	python train.py --input_dir <training_images_folder>

For more control over the training, refer to `config.py` which contains all the configurable settings to be used. You'll find comments there to help you out.

As mentioned above, the code in this repo runs faster than [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow). 

The following examples are run on Tesla V100 DGXS GPU.

##### Original Training

![Old code](https://i.ibb.co/9WGySjc/old-code.png)

##### Training using code in this repo

![New code](https://i.ibb.co/wcVPh3k/new-code.png)

#### 2. Generating samples

For generating output samples using your trained model, you should follow this pattern.

- Create folder `inputs` inside folder `test_data` inside the project root. (Note: You can change this in the `config.py` file)
- Put your input images inside the `inputs` folder and run `test.py`.
- You can specify the `--checkpoint` flag to point to the folder where model checkpoints are saved.


### Contributors

1. [Pranav Gajjewar](https://in.linkedin.com/in/pranav-gajjewar-a9647a137)
2. [Omkar Thawakar](https://github.com/OmkarThawakar)


