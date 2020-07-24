import tensorflow as tf
from tensorflow.python.platform import gfile
import torch
import numpy as np
import os
import cv2
import argparse
import time
from multiprocessing import Pool

from YOLOv1.modules import TinyYOLOv1Backbone, YOLOv1Backbone

np.random.seed(520)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="Transform Tensorflow weights(.pb) to Pytorch weights(.pth).")
    parser.add_argument('--load_path', type=str, default='',
                        help='Save path for Tensorflow weights(.pb).')
    parser.add_argument('--save_path', type=str, default='',
                        help='Save path for Pytorch weights(.pth).')
    parser.add_argument('--model_name', type=str, default='',
                        help='Name of the model. '
                             'Optional: yolov1, yolov1-tiny.')
    parser.add_argument('--num_processes', type=int, default=0,
                        help='Number of processes. '
                             'If zero, multiprocessing will not be used.')
    return parser.parse_args().__dict__


def extract_weights(variable_list, ops):
    for i in range(len(variable_list)):
        output = ops[variable_list[i]].outputs[0]
        variable_list[i] = sess.run(output)


if __name__ == '__main__':
    args = parse_args()
    load_path = args["load_path"]
    save_path = args["save_path"]
    model_name = args["model_name"].lower()
    num_processes = args["num_processes"]
    if load_path == "" or save_path == "" or model_name == "":
        exit(-1)

    with tf.Session() as sess:
        """load model"""
        with gfile.FastGFile(load_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        sess.run(tf.global_variables_initializer())

        """display the model with tensorboard"""
        # summary_writer = tf.summary.FileWriter('log/')
        # summary_writer.add_graph(sess.graph)
        # summary_writer.close()

        """output shape of each layer"""
        inp = np.random.random([1, 448, 448, 3])
        ops = sess.graph.get_operations()
        for i in range(len(ops) - 1, -1, -1):
            if "init" in ops[i].name:
                ops.pop(i)
        for op in ops:
            print(op.name)
            print(op.outputs[0].shape)

        """extract weights and biases"""
        if model_name == "yolov1-tiny":
            conv_weights = [3, 19, 35, 51, 67, 83, 99, 114]
            bn_means = [i + 2 for i in conv_weights]
            bn_stds = [i + 2 for i in bn_means]
            bn_gammas = [i + 2 for i in bn_stds]
            bn_betas = [i + 2 for i in bn_gammas]
            fc_weights = [137]
            fc_biases = [138]
            local2d_weights = []
        elif model_name == "yolov1":
            conv_weights = [3, 19, 35, 50, 65, 80, 96, 111, 126, 141, 156, 171, 186, 201, 216, 231, 247, 262, 277, 292,
                            307, 322, 337, 352]
            bn_means = [i + 2 for i in conv_weights]
            bn_stds = [i + 2 for i in bn_means]
            bn_gammas = [i + 2 for i in bn_stds]
            bn_betas = [i + 2 for i in bn_gammas]
            fc_weights = [690]
            fc_biases = [691]
            # conv_weights = []
            # bn_means = []
            # bn_stds = []
            # bn_gammas = []
            # bn_betas = []
            # fc_weights = []
            # fc_biases = []
            local2d_weights = [[371, 377, 383, 389, 395, 401, 407,
                                415, 421, 427, 433, 439, 445, 451,
                                459, 465, 471, 477, 483, 489, 495,
                                503, 509, 515, 521, 527, 533, 539,
                                547, 553, 559, 565, 571, 577, 583,
                                591, 597, 603, 609, 615, 621, 627,
                                635, 641, 647, 653, 659, 665, 671]]
            local2d_sizes = [7]

        print("Extracting weights... ", end="", flush=True)
        start_time = time.time()
        names = ["conv_weights",
                 "bn_means", "bn_stds", "bn_gammas", "bn_betas",
                 "fc_weights", "fc_biases",
                 "local2d_weights"]
        for name in names:
            if "local2d" in name:
                v_lists = locals()[name]
                for i in range(len(v_lists)):
                    extract_weights(v_lists[i], ops)
            else:
                extract_weights(locals()[name], ops)
        print("Done:", time.time() - start_time, "s")

        """feed weights"""
        if model_name == "yolov1-tiny":
            pth_model = TinyYOLOv1Backbone()
        elif model_name == "yolov1":
            pth_model = YOLOv1Backbone()
        pth_model.eval()
        print("Feeding weights... ", end="", flush=True)
        start_time = time.time()
        for conv_id, weights in enumerate(conv_weights):
            layer = getattr(pth_model, "conv" + str(conv_id + 1))
            layer.weight = torch.nn.Parameter(torch.tensor(weights).permute(3, 2, 0, 1), requires_grad=True)

        for bn_id, means in enumerate(bn_means):
            layer = getattr(pth_model, "batch_norm" + str(bn_id + 1))
            layer.running_mean = torch.nn.Parameter(torch.tensor(means), requires_grad=False)
        for bn_id, stds in enumerate(bn_stds):
            layer = getattr(pth_model, "batch_norm" + str(bn_id + 1))
            layer.running_var = torch.nn.Parameter(torch.tensor(stds) ** 2, requires_grad=False)
        for bn_id, gammas in enumerate(bn_gammas):
            layer = getattr(pth_model, "batch_norm" + str(bn_id + 1))
            layer.weight = torch.nn.Parameter(torch.tensor(gammas), requires_grad=True)
        for bn_id, betas in enumerate(bn_betas):
            layer = getattr(pth_model, "batch_norm" + str(bn_id + 1))
            layer.bias = torch.nn.Parameter(torch.tensor(betas), requires_grad=True)

        for fc_id, weights in enumerate(fc_weights):
            layer = getattr(pth_model, "fc" + str(fc_id + 1))
            layer.weight = torch.nn.Parameter(torch.tensor(weights).permute(1, 0), requires_grad=True)
        for fc_id, biases in enumerate(fc_biases):
            layer = getattr(pth_model, "fc" + str(fc_id + 1))
            layer.bias = torch.nn.Parameter(torch.tensor(biases), requires_grad=True)

        for local2d_id, weights in enumerate(local2d_weights):
            weights = torch.tensor(np.array(weights))
            weights = weights.view(local2d_sizes[local2d_id], local2d_sizes[local2d_id], *weights.size()[1:])
            weights = weights.permute(5, 4, 0, 1, 2, 3)
            layer = getattr(pth_model, "local" + str(local2d_id + 1))
            layer.weight = torch.nn.Parameter(weights, requires_grad=True)

        print("Done:", time.time() - start_time, "s")

        # torch.save(pth_model, save_path)
        """validate"""
        inp = cv2.imread("data/dog.jpg").copy()
        inp = inp[:, :, ::-1]
        inp = inp / 255.
        inp = inp.copy()
        inp = np.expand_dims(inp, 0)

        outputs_pth = {}


        def get_output_hook(name):
            def hook(model, input, output):
                print(model)
                outputs_pth[name] = output

            return hook


        for name, module in pth_model.named_children():
            module.register_forward_hook(get_output_hook(name))
        output_pth = pth_model(torch.from_numpy(inp))

        tff = lambda x: sess.run(sess.graph.get_tensor_by_name(x), feed_dict={"input:0": inp})
        # print(
        #     outputs_pth['conv1'].detach().numpy().swapaxes(1, 3).swapaxes(1, 2) -
        #     tff("0-convolutional:0")
        # )
        # print(
        #     (outputs_pth['conv1'].detach().numpy().swapaxes(1, 3).swapaxes(1, 2) - tff("sub/y:0")) /
        #     tff("truediv/y:0") * tff("mul/y:0") + tff("BiasAdd/bias:0") -
        #     tff("BiasAdd:0")
        # )
        # print(
        #     outputs_pth['conv3'].detach().numpy().swapaxes(1, 3).swapaxes(1, 2) -
        #     tff("6-convolutional:0")
        # )
        # print(
        #     outputs_pth['conv8'].detach().numpy().swapaxes(1, 3).swapaxes(1, 2) -
        #     tff("20-convolutional:0")
        # )
        print(
            outputs_pth['fc1'].detach().numpy() -
            tff("output:0")
        )
        print([[-3.1515956e-06, 7.7709556e-06, -1.6540289e-06, 8.5234642e-06, 6.6757202e-06, -5.9604645e-08]])

        print()
