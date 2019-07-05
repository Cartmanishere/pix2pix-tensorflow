# Pix2Pix Tensorflow

This is a general purpose implementation of the pix2pix algorithm for image to image translation. 
This algorithm is based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

Code in this repo has been heavily borrowed from [this](https://github.com/affinelayer/pix2pix-tensorflow) implementaion
of the pix2pix tensorflow. But the linked code is not easy to use as I have experienced it first hand while 
working on a related project.

This repo is an attempt to make the pix2pix implementation easily approachable for training and testing. The provided code 
also trains faster by removing some of the unnecessary operations. As such, there is some difference between how to define 
input and output in the linked repo and this one.

A guide on how to use this repo for image to image translation is provided.

---

### 1. Dependencies:

You should have the following dependencies installed in your system to run this code.

- `python 3` runtime environment
- `tensorflow==1.10.0`
