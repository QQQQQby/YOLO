# coding: utf-8

import os
import random
from xml.dom import minidom
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np


class DataLoader:
    """The base class for reading data, the data is stored in the form of a list"""

    def __init__(self, path):
        """Root directory of the dataset"""
        self.__path = path

    def get_data_train(self):
        raise NotImplementedError()

    def get_data_eval(self):
        raise NotImplementedError()

    def get_data_test(self):
        raise NotImplementedError()

    def get_path(self):
        return self.__path


class VOCDataLoader(DataLoader):
    def __init__(self, path, train_prop=0.8, num_processes=0, do_shuffle=False, preload=True):
        super(VOCDataLoader, self).__init__(path)
        image_file_names = sorted(os.listdir(os.path.join(self.get_path(), "JPEGImages")))
        image_file_names = [name[:-4] for name in image_file_names]
        annotation_file_names = sorted(os.listdir(os.path.join(self.get_path(), "Annotations")))
        annotation_file_names = [name[:-4] for name in annotation_file_names]

        if image_file_names != annotation_file_names:
            # Make sure every image has a corresponding annotation file.
            print("Warning: Incomplete dataset.")
        if do_shuffle:
            random.shuffle(image_file_names)

        split_index = int(train_prop * len(image_file_names))
        # self.train_file_names = image_file_names[:split_index]
        self.train_file_names = image_file_names[:split_index]
        # self.eval_file_names = image_file_names[split_index:]
        self.eval_file_names = image_file_names[split_index:]

        self.num_processes = num_processes
        self.preload = preload

        self.data_train = None
        self.data_eval = None

    def get_all_data(self):
        return self.get_data_train() + self.get_data_eval()

    def get_data_train(self):
        """Use multiprocessing to read data"""
        if self.data_train is not None:
            return self.data_train
        image_object_paths = []
        for name in self.train_file_names:
            image_object_paths.append(
                [os.path.join(self.get_path(), "JPEGImages", name + '.jpg'),
                 os.path.join(self.get_path(), "Annotations", name + '.xml')]
            )
        res = []
        process_bar = tqdm(range(len(self.train_file_names)))
        if self.num_processes == 0:
            for paths in image_object_paths:
                # print(paths)
                res.append(self.read_image_and_objects(paths))
                process_bar.update()
        else:
            with Pool(self.num_processes) as p:
                for data in p.imap(self.read_image_and_objects, image_object_paths):
                    res.append(data)
                    process_bar.update()
                p.close()
        process_bar.close()
        self.data_train = res
        return res

    def get_data_eval(self):
        if self.data_eval is not None:
            return self.data_eval
        image_object_paths = []
        for name in self.eval_file_names:
            image_object_paths.append(
                (os.path.join(self.get_path(), "JPEGImages", name + '.jpg'),
                 os.path.join(self.get_path(), "Annotations", name + '.xml'))
            )
        res = []
        process_bar = tqdm(range(len(self.eval_file_names)))
        if self.num_processes == 0:
            for paths in image_object_paths:
                # print(paths)
                res.append(self.read_image_and_objects(paths))
                process_bar.update()
        else:
            with Pool(self.num_processes) as p:
                for data in p.imap(self.read_image_and_objects, image_object_paths):
                    res.append(data)
                    process_bar.update()
                p.close()
        process_bar.close()
        self.data_eval = res
        return res

    def read_image_and_objects(self, image_object_path):
        return [
            self.read_image_array(image_object_path[0]) if self.preload else image_object_path[0],
            self.read_objects(image_object_path[1])
        ]

    @classmethod
    def read_image_array(self, path):
        array = cv2.imread(path)
        return array[:, :, ::-1]

    @classmethod
    def read_objects(cls, path):
        res = []
        dom = minidom.parse(path)

        # size = dom.getElementsByTagName('size')[0]
        # w_abs = eval(size.getElementsByTagName('width')[0].firstChild.data)
        # h_abs = eval(size.getElementsByTagName('height')[0].firstChild.data)

        objects = dom.getElementsByTagName('object')
        for o in objects:
            name = o.getElementsByTagName('name')[0].firstChild.data
            box = o.getElementsByTagName('bndbox')[0]
            xmin = eval(box.getElementsByTagName('xmin')[0].firstChild.data)  # [1, width]
            ymin = eval(box.getElementsByTagName('ymin')[0].firstChild.data)  # [1, height]
            xmax = eval(box.getElementsByTagName('xmax')[0].firstChild.data)  # [1, width]
            ymax = eval(box.getElementsByTagName('ymax')[0].firstChild.data)  # [1, height]
            x = int(((xmin + xmax) / 2 - 1))  # [0, width-1]
            y = int(((ymin + ymax) / 2 - 1))  # [0, height-1]
            w = int((xmax - xmin))
            h = int((ymax - ymin))
            res.append(dict(name=name, x=x, y=y, w=w, h=h))
        return res


def read_classes(path):
    with open(path, "r", encoding='utf-8') as f:
        return f.read().strip().split("\n")


def get_color_dict(classes, color_path):
    color_dict = {}
    with open(color_path, "r") as f:
        for c in classes:
            color_dict[c] = tuple([int(t) for t in f.readline().split(" ")])
    return color_dict


def read_anchors(path):
    with open(path) as f:
        l = [float(v) for v in f.readline().split(",")]
    return np.array([[l[i], l[i + 1]] for i in range(0, len(l), 2)])
