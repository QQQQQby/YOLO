# Pytorch Implement of YOLO 

## Introduction

The full name of **YOLO** is **Y**ou **O**nly **L**ook **O**nce. It is a popular model with high speed and accuracy used for **Object Detection**. You can learn more in [the official website](https://pjreddie.com/darknet/yolo/).

## Dataset

[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

## Requirements

* Python>=3.5
* Pytorch>=1.4
* OpenCV
* moviepy
* scipy
* PIL

## Detect

You can run:

```
python detect.py
    --model_load_path=models/yolov3.pth
    --class_path=data/coco-ch.names
    --input_path=data/dog.jpg
    --output_path=data/dog_pred.jpg
    --device_ids=0
```

Now enjoy!

All usages and optional arguments:
```
 usage: detect.py [-h] [--model_load_path MODEL_LOAD_PATH]
                 [--class_path CLASS_PATH] [--color_path COLOR_PATH]
                 [--anchor_path ANCHOR_PATH] [--input_path INPUT_PATH]
                 [--output_path OUTPUT_PATH] [--not_show]
                 [--score_threshold SCORE_THRESHOLD]
                 [--iou_threshold IOU_THRESHOLD] [--device_ids DEVICE_IDS]
                 [--num_processes NUM_PROCESSES]

Object detection.

optional arguments:
  -h, --help            show this help message and exit
  --model_load_path MODEL_LOAD_PATH
                        Input path to models.
  --class_path CLASS_PATH
                        Path to a file to store names and colors of the
                        classes.
  --color_path COLOR_PATH
                        Path to a file which stores colors.
  --anchor_path ANCHOR_PATH
                        Input path to anchors.
  --input_path INPUT_PATH
                        Path to the file used for detection. If zero, camera
                        on your computer will be used.
  --output_path OUTPUT_PATH
                        Path to the output image or video. If Empty, the
                        predicted image will not be saved.
  --not_show            Whether not to show predictions.
  --score_threshold SCORE_THRESHOLD
                        Threshold of score(IOU * P(Object)).
  --iou_threshold IOU_THRESHOLD
                        Threshold of IOU used for calculation of NMS.
  --device_ids DEVICE_IDS
                        Device ids. Should be seperated by commas. -1 means
                        cpu.
  --num_processes NUM_PROCESSES
                        number of processes.
```

## Run with flask

```
python run_flask.py
```

## Transform .weights to .pth

We firstly use `.weights` and `.cfg` files to generate and save a Tensorflow model. The table below shows how to do this.

| Model               | repo                                | outputs                                       |
| ------------------- | ----------------------------------- | --------------------------------------------- |
| yolov1, yolov1-tiny | https://github.com/thtrieu/darkflow | a `.pb` file and a `.meta` file               |
| yolov3              | https://github.com/jinyu121/DW2TF   | 3 `.ckpt` files and a file named `checkpoint` |

Then we use these files to generate a Pytorch model by running `pb2pth.py`.

## Thanks

[darkflow](https://github.com/thtrieu/darkflow)

[DW2TF](https://github.com/jinyu121/DW2TF)