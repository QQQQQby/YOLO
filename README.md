# Pytorch Implement of YOLO 

## Introduction

The full name of **YOLO** is **Y**ou **O**nly **L**ook **O**nce. It is a popular model with high speed and accuracy used for **Object Detection**. You can learn more in [the official website](https://pjreddie.com/darknet/yolo/).

## Dataset

[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

## Requirements

* Python>=3.5
* Pytorch>=1.4
* OpenCV

## How to run

You can run:

```
python run_classifier.py
    --model_load_path=%MODEL_PATH%
    --video_detect_path=%VIDEO_PATH%
    --use_cuda
```

Now enjoy!

And there are some other optional arguments:
```
  -h, --help            show this help message and exit
  --image_detect_path IMAGE_DETECT_PATH
                        Image path for detection. If empty, the detection will
                        not perform.
  --video_detect_path VIDEO_DETECT_PATH
                        Image path for detection. If zero, OpenCV will predict
                        through camera. If empty, the detection will not
                        perform.
  --dataset_path DATASET_PATH
                        Dataset path.
  --preload             Whether to preload the dataset.
  --model_load_path MODEL_LOAD_PATH
                        Input path for models.
  --model_name {yolov1,yolov1-tiny}
                        Model type. optional models: yolov1(default),
                        yolov1-tiny. Not required when the loading path of the
                        model is specified.
  --class_path CLASS_PATH
                        Path for a file to store names and colors of the classes.
  --graph_save_dir GRAPH_SAVE_DIR
                        Output directory for the graph of the model. If empty,
                        graph will not be saved.
  --use_cuda            Whether to use cuda to run the model.
  --do_train            Whether to train the model on dataset.
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size of train set.
  --num_epochs NUM_EPOCHS
                        Number of epochs.
  --lr LR               Learning rate.
  --momentum MOMENTUM   Momentum of optimizer.
  --lambda_coord LAMBDA_COORD
                        Lambda of coordinates.
  --lambda_noobj LAMBDA_NOOBJ
                        Lambda with no objects.
  --clip_max_norm CLIP_MAX_NORM
                        Max norm of the gradients. If zero, the gradients will
                        not be clipped.
  --model_save_dir MODEL_SAVE_DIR
                        Output directory for the model. When empty, the model
                        will not be saved
  --do_eval             Whether to evaluate the model on dataset.
  --eval_batch_size EVAL_BATCH_SIZE
                        Batch size of evaluation set.
  --score_threshold SCORE_THRESHOLD
                        Threshold of score(IOU * P(Object)).
  --iou_threshold IOU_THRESHOLD
                        Threshold of IOU used for calculation of NMS.
  --iou_thresholds_mmAP IOU_THRESHOLDS_MMAP
                        Thresholds of IOU used for calculation of mmAP.
  --do_test             Whether to test the model.
  --test_batch_size TEST_BATCH_SIZE
                        Batch size of test set.
```