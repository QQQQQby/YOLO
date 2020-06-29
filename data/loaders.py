import os
from xml.dom import minidom
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class DataLoader:
    """读取数据的基类，数据以列表形式存储"""

    def __init__(self, path):
        """定义数据集根目录"""
        self.__path = path

    def get_data_train(self):
        """获取训练集数据"""
        raise NotImplementedError()

    def get_data_dev(self):
        """获取开发集数据"""
        raise NotImplementedError()

    def get_data_test(self):
        """获取测试集数据"""
        raise NotImplementedError()

    def get_labels(self):
        """获取所有标签"""
        raise NotImplementedError()

    def get_path(self):
        return self.__path


class VOC2012Loader(DataLoader):
    def __init__(self, path, train_prop=0.8):
        super(VOC2012Loader, self).__init__(path)
        file_names = os.listdir(os.path.join(self.get_path(), "VOC2012", "train", "JPEGImages"))
        file_names = [name.replace('.jpg', '') for name in file_names]

        file_names = file_names[:200]

        split_index = int(train_prop * len(file_names))
        self.train_file_names = file_names[:split_index]
        self.dev_file_names = file_names[split_index:]

    def get_data_train(self):
        res = []
        for name in tqdm(self.train_file_names):
            image_path = os.path.join(self.get_path(), "VOC2012", "train", "JPEGImages", name + '.jpg')
            object_path = os.path.join(self.get_path(), "VOC2012", "train", "Annotations", name + '.xml')
            image = self.read_image_array(image_path)
            objects = self.read_objects(object_path)
            res.append([image, objects])
        return res

    def get_data_dev(self):
        res = []
        for name in tqdm(self.dev_file_names):
            image_path = os.path.join(self.get_path(), "VOC2012", "train", "JPEGImages", name + '.jpg')
            object_path = os.path.join(self.get_path(), "VOC2012", "train", "Annotations", name + '.xml')
            image = self.read_image_array(image_path)
            objects = self.read_objects(object_path)
            res.append([image, objects])
        return res

    def get_data_test(self):
        pass

    def get_labels(self):
        return ["person",
                "bird", "cat", "cow", "dog", "horse", "sheep",
                "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

    @classmethod
    def read_image_array(self, path):
        array = cv2.imread(path)
        array = cv2.resize(array, (448, 448))
        return array[:, :, ::-1]

    @classmethod
    def read_objects(cls, path):
        res = []
        dom = minidom.parse(path)

        size = dom.getElementsByTagName('size')[0]
        w_abs = eval(size.getElementsByTagName('width')[0].firstChild.data)
        h_abs = eval(size.getElementsByTagName('height')[0].firstChild.data)

        objects = dom.getElementsByTagName('object')
        for o in objects:
            name = o.getElementsByTagName('name')[0].firstChild.data
            box = o.getElementsByTagName('bndbox')[0]
            xmin = eval(box.getElementsByTagName('xmin')[0].firstChild.data)
            ymin = eval(box.getElementsByTagName('ymin')[0].firstChild.data)
            xmax = eval(box.getElementsByTagName('xmax')[0].firstChild.data)
            ymax = eval(box.getElementsByTagName('ymax')[0].firstChild.data)
            x = int(((xmin + xmax) / 2 - 1) / (w_abs - 1) * (448 - 1))
            y = int(((ymin + ymax) / 2 - 1) / (h_abs - 1) * (448 - 1))
            w = int((xmax - xmin) / w_abs * 448)
            h = int((ymax - ymin) / h_abs * 448)
            res.append(dict(name=name, x=x, y=y, w=w, h=h))
        return res


if __name__ == '__main__':
    loader = VOC2012Loader('G:/DataSets')
