# LeagueAI
Implementation of an A.I. Player for the videogame League of Legends based on Image Recognition using PyTorch

Demo video of Tensorflow implementation from 2017: https://www.youtube.com/watch?v=KRWFCaXfOTk

## TODO
- Create script to automatically split data in train and test set

## Currently Detectable Objects

## Missing Objects
- Add Red inhibitors, nexus, super minions
- Add Blue towers, nexus, inhibitors, minions

## Abstract
The task is to create an agent that is able to play 3rd person massive multiplayer online battle arena games (MOBA) like League of Legends, Dota 2 and Heroes of the Storm with the same input as a human player.
Since League of Legends does not provide an interface to the game, object detection is used.
In this project a python implementation of Yolo v3 object detector and a way of randomly generating an infine amout of training data is introduced.

## Object detection
TODO: Describe the object detector was implemented and can be used

## The LeagueAI Dataset
Creating large datasets from scratch can be very work intensive. For the first implementation of the LeageAI about 700 hand labeled pictures were used. Labeling 700 pictures took about 4 days of work and only included 4 game objects (1 champion model, allied and enemy minions and enemy towers). Therefore, the new dataset was created by automatically generating training data based on 3D models extracted from the game.

1. Obtaining champion and minion models form 3D models
To obtain the image data I used the online League of Legends model viewer from https://teemo.gg/model-viewer. For each ingame object and each animation I recorded a short video clip while rotating the 3D model.
Next I used the pyFrameExporter.py script to extract individual pictures from the clips.
For each of the 
For the minions I used Adobe After Effects to remove the background

2. Combining the masked and cropped images with game background 
In order to generate a large amount of training data that cover all regions of the game map, generated a series of 200 screenshots from all over the map.
Then the masked and cropped images are randomly combined with the map screenshots.
Since we place the images using a script it is possible to obtain the objects position in the image and thus automatically generate a label for it.

The result is a dataset of XXXXXX labeled images that can be increased anytime by randomly placing the objects on the map.

## Extracting health information
