humble-yolo is a minimal implementation of YOLO v1 I wrote to learn about the amazing YOLO algorithm.

To test it run :

1. generate-dataset.py to generate data
2. main.py --train --epoch 100 for training the network

You should see a list of images with bounding boxes. The first 10 images are test data not used for training. You can evaluate the performance of the network on those. The remaining images have been used for the training.

main.py saves weights when it complete training. If you want to run the network without training and just see the result, running main.py alone will load last weights and redisplay results.
