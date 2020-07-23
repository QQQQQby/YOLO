# coding: utf-8

import os
from xml.dom import minidom
import cv2
from tqdm import tqdm
from multiprocessing import Pool


class DataLoader:
    """The base class for reading data, the data is stored in the form of a list"""

    def __init__(self, path):
        """Define root directory of the dataset"""
        self.__path = path

    def get_data_train(self):
        raise NotImplementedError()

    def get_data_eval(self):
        raise NotImplementedError()

    def get_data_test(self):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    def get_path(self):
        return self.__path


class VOC2012Loader(DataLoader):
    def __init__(self, path, train_prop=0.8, num_processes=4):
        super(VOC2012Loader, self).__init__(path)
        file_names = os.listdir(os.path.join(self.get_path(), "VOC2012", "train", "JPEGImages"))
        file_names = [name.replace('.jpg', '') for name in file_names]

        # file_names = file_names[:2000]

        split_index = int(train_prop * len(file_names))
        # self.train_file_names = file_names[:split_index]
        self.train_file_names = file_names[:50]
        # self.eval_file_names = file_names[split_index:]
        self.eval_file_names = file_names[:]

        self.num_processes = num_processes

    def get_data_train(self):
        """Use multiprocessing to read data"""
        image_object_paths = []
        for name in self.train_file_names:
            image_object_paths.append(
                [os.path.join(self.get_path(), "VOC2012", "train", "JPEGImages", name + '.jpg'),
                 os.path.join(self.get_path(), "VOC2012", "train", "Annotations", name + '.xml')]
            )
        res = []
        process_bar = tqdm(range(len(self.train_file_names)))
        with Pool(self.num_processes) as p:
            for data in p.imap(self.read_image_and_objects, image_object_paths):
                res.append(data)
                process_bar.update()
            p.close()
        process_bar.close()
        return res

        # res = [None for i in range(len(self.dev_file_names))]
        #
        # processes = []
        # length = int(float(len(self.dev_file_names)) / float(self.num_processes))
        # indices = [int(round(i * length)) for i in range(self.num_processes)]
        # indices.append(len(self.dev_file_names))
        #
        # train_file_name_sublists = [self.dev_file_names[indices[i]:indices[i + 1]]
        #                             for i in range(self.num_processes)]
        #
        # for i in range(self.num_processes):
        #     image_paths = []
        #     object_paths = []
        #     for j in range(len(train_file_name_sublists[i])):
        #         image_paths.append(os.path.join(
        #             self.get_path(), "VOC2012", "train", "JPEGImages",
        #             train_file_name_sublists[i][j] + '.jpg'
        #         ))
        #         object_paths.append(os.path.join(
        #             self.get_path(), "VOC2012", "train", "Annotations",
        #             train_file_name_sublists[i][j] + '.xml'
        #         ))
        #     processes.append(Process(
        #         target=self.read_data_to_list,
        #         args=(res, indices[i], indices[i + 1], image_paths, object_paths))
        #     )
        # for p in processes:
        #     p.start()
        # for p in processes:
        #     p.join()
        # return res

    # def read_data_to_list(self, res, start, end, image_paths, object_paths):
    #     for path_i, list_i in enumerate(range(start, end)):
    #         res[list_i] = (
    #             self.read_image_array(image_paths[path_i]),
    #             self.read_objects(object_paths[path_i])
    #         )

    def get_data_eval(self):
        image_object_paths = []
        for name in self.eval_file_names:
            image_object_paths.append(
                (os.path.join(self.get_path(), "VOC2012", "train", "JPEGImages", name + '.jpg'),
                 os.path.join(self.get_path(), "VOC2012", "train", "Annotations", name + '.xml'))
            )
        res = []
        process_bar = tqdm(range(len(self.eval_file_names)))
        with Pool(self.num_processes) as p:
            for data in p.imap(self.read_image_and_objects, image_object_paths):
                res.append(data)
                process_bar.update()
            p.close()
        process_bar.close()
        return res

    def get_data_test(self):
        pass

    def get_labels(self):
        return sorted(["person",
                       "bird", "cat", "cow", "dog", "horse", "sheep",
                       "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                       "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"])

    def read_image_and_objects(self, image_object_path):
        return [
            self.read_image_array(image_object_path[0]),
            self.read_objects(image_object_path[1])
        ]

    @classmethod
    def read_image_array(self, path):
        # array = cv2.imread(path)
        # array = cv2.resize(array, (448, 448))
        # return array[:, :, ::-1]
        return path

    @classmethod
    def read_objects(cls, path):
        res = []
        dom = minidom.parse(path)

        size = dom.getElementsByTagName('size')[0]
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


if __name__ == '__main__':
    loader = VOC2012Loader('G:/DataSets')
