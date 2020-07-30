import tensorflow as tf
from tensorflow.python.platform import gfile
import torch
from torch.utils import tensorboard
import numpy as np
import os
import cv2
import argparse
import time

from YOLOv1.modules import TinyYOLOv1Backbone, YOLOv1Backbone
from YOLOv3.modules import YOLOv3Backbone

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_args():
    parser = argparse.ArgumentParser(description="Transform Tensorflow weights(.pb) to Pytorch weights(.pth).")
    parser.add_argument('--load_path', type=str, default='',
                        help='Save path for Tensorflow weights(.pb or .ckpt).')
    parser.add_argument('--save_path', type=str, default='',
                        help='Save path for Pytorch weights(.pth).')
    # parser.add_argument('--valid_image_path', type=str, default='',
    #                     help='Path of image for validation.'
    #                          'If empty, the validation will not proceed.')
    parser.add_argument('--model_name', type=str, default='',
                        help='Name of the model. ',
                        choices=["yolov1", "yolov1-tiny", "yolov3"])
    return parser.parse_args().__dict__


def extract_weights(variable_list, ops):
    for i in range(len(variable_list)):
        if variable_list[i] is None:
            continue
        output = ops[variable_list[i]].outputs[0]
        variable_list[i] = sess.run(output)


if __name__ == '__main__':
    args = parse_args()
    load_path = args["load_path"]
    save_path = args["save_path"]
    model_name = args["model_name"]
    if load_path == "" or save_path == "":
        exit(-1)

    with tf.Session() as sess:
        """load model"""
        if load_path.endswith(".pb"):
            with gfile.FastGFile(load_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
        elif load_path.endswith(".ckpt"):
            load_dir = os.path.dirname(load_path)
            sess.graph.as_default()
            saver = tf.train.import_meta_graph(load_path + ".meta")
            saver.restore(sess, tf.train.latest_checkpoint(load_dir))

        """display the model with tensorboard"""
        # summary_writer = tf.summary.FileWriter('log/')
        # summary_writer.add_graph(sess.graph)
        # summary_writer.close()

        """output shape of each layer"""
        # inp = np.random.random([1, 448, 448, 3])
        ops = sess.graph.get_operations()
        for i in range(len(ops) - 1, -1, -1):
            if "init" in ops[i].name:
                ops.pop(i)
        for op in ops:
            print(op.name)
            # print(op.outputs[0].shape)

        """extract weights and biases"""
        if model_name == "yolov1-tiny":
            conv_weights = [3, 19, 35, 51, 67, 83, 99, 114]
            conv_biases = [None for i in range(len(conv_weights))]

            bn_means = [i + 2 for i in conv_weights]
            bn_stds = [i + 2 for i in bn_means]
            bn_vars = [None for i in bn_stds]
            bn_gammas = [i + 2 for i in bn_stds]
            bn_betas = [i + 2 for i in bn_gammas]

            fc_weights = [137]
            fc_biases = [138]
        elif model_name == "yolov1":
            conv_weights = [3, 19, 35, 50, 65, 80, 96, 111, 126, 141, 156, 171, 186, 201, 216, 231, 247, 262, 277, 292,
                            307, 322, 337, 352]
            conv_biases = [None for i in range(len(conv_weights))]

            bn_means = [i + 2 for i in conv_weights]
            bn_stds = [i + 2 for i in bn_means]
            bn_vars = [None for i in bn_stds]
            bn_gammas = [i + 2 for i in bn_stds]
            bn_betas = [i + 2 for i in bn_gammas]

            fc_weights = [690]
            fc_biases = [691]

            local2d_weights = [[371, 377, 383, 389, 395, 401, 407,
                                415, 421, 427, 433, 439, 445, 451,
                                459, 465, 471, 477, 483, 489, 495,
                                503, 509, 515, 521, 527, 533, 539,
                                547, 553, 559, 565, 571, 577, 583,
                                591, 597, 603, 609, 615, 621, 627,
                                635, 641, 647, 653, 659, 665, 671]]
            local2d_sizes = [7]
        elif model_name == "yolov3":
            conv_weights = []
            conv_biases = []
            bn_means = []
            bn_vars = []
            bn_gammas = []
            bn_betas = []
            for i, op in enumerate(ops):
                name = op.name
                if name.endswith("kernel/read"):
                    conv_weights.append(i)
                    conv_biases.append(None)

                elif name.endswith("bias/read"):
                    conv_biases[-1] = i
                elif name.endswith("moving_mean"):
                    bn_means.append(i)
                elif name.endswith("moving_variance"):
                    bn_vars.append(i)
                elif name.endswith("gamma"):
                    bn_gammas.append(i)
                elif name.endswith("beta"):
                    bn_betas.append(i)
            bn_stds = [None for i in bn_vars]

        print("Extracting weights... ", end="", flush=True)
        start_time = time.time()
        names = ["conv_weights", "conv_biases",
                 "bn_means", "bn_stds", "bn_vars", "bn_gammas", "bn_betas",
                 "fc_weights", "fc_biases",
                 "local2d_weights"]

        for name in names:
            locals()[name] = locals().get(name, [])
            v_list = locals()[name]
            if "local2d" in name:
                for i in range(len(v_list)):
                    extract_weights(v_list[i], ops)
            else:
                extract_weights(v_list, ops)
        print("Done:", time.time() - start_time, "s")

        if model_name == "yolov1-tiny":
            pth_model = TinyYOLOv1Backbone()
        elif model_name == "yolov1":
            pth_model = YOLOv1Backbone()
        elif model_name == "yolov3":
            pth_model = YOLOv3Backbone()
        print(pth_model)
        pth_model.eval()

        print("Feeding weights... ", end="", flush=True)
        start_time = time.time()
        for conv_id in range(len(conv_weights)):
            layer = getattr(pth_model, "conv" + str(conv_id + 1))
            layer.weight = torch.nn.Parameter(
                torch.tensor(conv_weights[conv_id]).permute(3, 2, 0, 1), requires_grad=True)
            if conv_biases[conv_id] is not None:
                layer.bias = torch.nn.Parameter(torch.tensor(conv_biases[conv_id]), requires_grad=True)

        for bn_id in range(len(bn_means)):
            layer = getattr(pth_model, "batch_norm" + str(bn_id + 1))
            # if bn_means[bn_id] is None:
            #     continue
            layer.running_mean = torch.nn.Parameter(torch.tensor(bn_means[bn_id]), requires_grad=False)
            if bn_vars[bn_id] is not None:
                layer.running_var = torch.nn.Parameter(torch.tensor(bn_vars[bn_id]), requires_grad=False)
            elif bn_stds[bn_id] is not None:
                layer.running_var = torch.nn.Parameter(torch.tensor(bn_stds[bn_id] ** 2), requires_grad=False)
            layer.weight = torch.nn.Parameter(torch.tensor(bn_gammas[bn_id]), requires_grad=True)
            layer.bias = torch.nn.Parameter(torch.tensor(bn_betas[bn_id]), requires_grad=True)

        for fc_id in range(len(fc_weights)):
            layer = getattr(pth_model, "fc" + str(fc_id + 1))
            layer.weight = torch.nn.Parameter(torch.tensor(fc_weights[fc_id]).permute(1, 0), requires_grad=True)
            layer.bias = torch.nn.Parameter(torch.tensor(fc_biases[fc_id]), requires_grad=True)

        for local2d_id, weight in enumerate(local2d_weights):
            weight = torch.tensor(np.array(weight))
            weight = weight.view(local2d_sizes[local2d_id], local2d_sizes[local2d_id], *weight.size()[1:])
            weight = weight.permute(5, 4, 0, 1, 2, 3)
            layer = getattr(pth_model, "local" + str(local2d_id + 1))
            layer.weight = torch.nn.Parameter(weight, requires_grad=True)

        print("Done:", time.time() - start_time, "s")

        # torch.save(pth_model, save_path)
        """validate"""
        outputs_pth = {}


        def get_output_hook(name):
            def hook(model, input, output):
                outputs_pth[name] = output

            return hook


        for name, module in pth_model.named_children():
            module.register_forward_hook(get_output_hook(name))

        inp = cv2.imread("data/dog.jpg").copy()
        inp = inp[:, :, ::-1]
        inp = inp / 255.
        inp = inp.copy()

        if "v1" in model_name:
            inp = np.expand_dims(inp, 0)
            output_pth = pth_model(torch.from_numpy(inp))
            print(
                outputs_pth['fc1'].detach().numpy() -
                sess.run(sess.graph.get_tensor_by_name("output:0"), feed_dict={"input:0": inp})
            )
        elif "v3" in model_name:
            inp = cv2.resize(inp, (608, 608))
            inp = np.expand_dims(inp, 0)
            output_pth = pth_model(torch.from_numpy(inp))
            o1, o2, o3, d = output_pth
            print(o1.shape)
            print(o2.shape)
            print(o3.shape)
            print(o1.permute(0, 2, 3, 1).detach().numpy() -
                  sess.run(sess.graph.get_tensor_by_name("yolov3/convolutional59/BiasAdd:0"),
                           feed_dict={"yolov3/net1:0": inp})
                  )

            # print(d["1"].permute(0, 2, 3, 1).detach().numpy() - sess.run(
            #     sess.graph.get_tensor_by_name("yolov3/convolutional1/Activation:0"),
            #     feed_dict={"yolov3/net1:0": inp}))
            # print(d["2"].permute(0, 2, 3, 1).detach().numpy() -sess.run(
            #                sess.graph.get_tensor_by_name("yolov3/convolutional2/Activation:0"),
            #                feed_dict={"yolov3/net1:0": inp}))
            # print(outputs_pth['conv2'].detach().permute(0, 2, 3, 1).numpy() -
            #       sess.run(sess.graph.get_tensor_by_name("yolov3/convolutional2/Conv2D:0"),
            #                feed_dict={"yolov3/net1:0": inp}))
            # print(outputs_pth['batch_norm1'].detach().permute(0, 2, 3, 1).numpy() -
            #       sess.run(sess.graph.get_tensor_by_name("yolov3/convolutional1/BatchNorm/FusedBatchNorm:0"),
            #                feed_dict={"yolov3/net1:0": inp}))

            print(o2.permute(0, 2, 3, 1).detach().numpy() -
                  sess.run(sess.graph.get_tensor_by_name("yolov3/convolutional67/BiasAdd:0"),
                           feed_dict={"yolov3/net1:0": inp})
                  )
            print(o3.permute(0, 2, 3, 1).detach().numpy() -
                  sess.run(sess.graph.get_tensor_by_name("yolov3/convolutional75/BiasAdd:0"),
                           feed_dict={"yolov3/net1:0": inp})
                  )
        # tff = lambda x: sess.run(sess.graph.get_tensor_by_name(x), feed_dict={"input:0": inp})
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

        print()
