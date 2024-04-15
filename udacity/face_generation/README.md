# Face-Generation

Face Generator Project is a part of Udacity Deep Learning Nanodegree.

# Goal

Generate new faces using Generative Adversarial Networks (GANs).
The model is trained on the CelebFaces Attributes Dataset (CelebA):


![Image of Training Set](./output/processed_face_data.png "Author : Arpit Dubey")

It generates new human faces that look like this:


![Image of Generated Faces](./output/Generated_faces.png "Author : Arpit Dubey")

## Recommended next steps

* Create a deeper model and use it to generate larger (say 128x128) images of faces.
* Read existing literature to see if you can use padding and normalization techniques to generate higher-resolution images.
* Implement a learning rate that evolves over time as they did in this [CycleGAN Github repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
* See if you can extend this model and use a CycleGAN to learn to swap different kinds of faces. For example, learn a mapping between faces that have and do not have eye/lip makeup, as they did in [this paper](https://gfx.cs.princeton.edu/pubs/Chang_2018_PAS/Chang-CVPR-2018.pdf).
