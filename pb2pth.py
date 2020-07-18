import tensorflow as tf
from tensorflow.python.platform import gfile
import torch
import numpy as np
import os
import cv2

from YOLOv1.modules import TinyYOLOv1Backbone

np.random.seed(520)
# torch.random.manual_seed(500)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# load_path = "../darkflow-master/built_graph/yolov1-tiny.pb"
load_path = "models/yolov1-tiny.pb"
save_path = "models/yolov1-tiny.pth"

with tf.Session() as sess:
    """load model"""
    with gfile.FastGFile(load_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    """display the model with tensorboard"""
    summary_wirter = tf.summary.FileWriter('log/', sess.graph)
    summary_wirter.add_graph(graph_def)

    """output shape of each layer"""
    inp = np.random.random([1, 448, 448, 3])

    # for op in sess.graph.get_operations():
    #     print(op.name)
    #     if "init" not in op.name:
    #         print(np.shape(sess.run(op.outputs[0], feed_dict={"input:0": inp})))
    #     else:
    #         print(np.shape(sess.run(op.outputs, feed_dict={"input:0": inp})))

    # print(sess.run(
    #     sess.graph.get_tensor_by_name("mul_3/x:0"),
    #     feed_dict={"input:0": inp}
    # ))
    #

    for op in sess.graph.get_operations():
        print(op.name)

    """extract weights and biases"""
    conv_weights = [3, 19, 35, 51, 67, 83, 99, 114]
    bn_means = [i + 2 for i in conv_weights]
    bn_stds = [i + 2 for i in bn_means]
    bn_gammas = [i + 2 for i in bn_stds]
    bn_betas = [i + 2 for i in bn_gammas]
    fc_weights = [137]
    fc_biases = [138]
    ops = sess.graph.get_operations()

    names = ["conv_weights", "bn_means", "bn_stds", "bn_gammas", "bn_betas", "fc_weights", "fc_biases"]
    for name in names:
        for i in range(len(locals()[name])):
            output = ops[locals()[name][i]].outputs[0]
            if name == "conv_weights":
                output = tf.transpose(output, [3, 2, 0, 1])
            elif name == "fc_weights":
                output = tf.transpose(output, [1, 0])
            locals()[name][i] = sess.run(output)

    """feed weights"""
    pth_model = TinyYOLOv1Backbone()
    pth_model.eval()
    for conv_id, weights in enumerate(conv_weights):
        layer = getattr(pth_model, "conv" + str(conv_id + 1))
        layer.weight = torch.nn.Parameter(torch.tensor(weights), requires_grad=True)

    for bn_id, means in enumerate(bn_means):
        layer = getattr(pth_model, "batch_norm" + str(bn_id + 1))
        layer.running_mean = torch.nn.Parameter(torch.tensor(means), requires_grad=False)
    for bn_id, stds in enumerate(bn_stds):
        layer = getattr(pth_model, "batch_norm" + str(bn_id + 1))
        layer.running_var = torch.nn.Parameter(torch.tensor(stds) ** 2, requires_grad=False)
    for bn_id, gammas in enumerate(bn_gammas):
        layer = getattr(pth_model, "batch_norm" + str(bn_id + 1))
        layer.weight = torch.nn.Parameter(torch.tensor(gammas), requires_grad=False)
    for bn_id, betas in enumerate(bn_betas):
        layer = getattr(pth_model, "batch_norm" + str(bn_id + 1))
        layer.bias = torch.nn.Parameter(torch.tensor(betas), requires_grad=False)

    for fc_id, weights in enumerate(fc_weights):
        layer = getattr(pth_model, "fc" + str(fc_id + 1))
        layer.weight = torch.nn.Parameter(torch.tensor(weights), requires_grad=True)
    for fc_id, biases in enumerate(fc_biases):
        layer = getattr(pth_model, "fc" + str(fc_id + 1))
        layer.bias = torch.nn.Parameter(torch.tensor(biases), requires_grad=True)

    torch.save(pth_model, save_path)
    """validate"""
    # inp = np.random.random([1, 448, 448, 3]) * 128
    # inp = np.array(inp, dtype=np.uint8)
    inp = cv2.imread("dog.jpg").copy()
    inp = inp[:, :, ::-1] / 255.

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
    print(
        outputs_pth['conv1'].detach().numpy().swapaxes(1, 3).swapaxes(1, 2) -
        tff("0-convolutional:0")
    )
    # print(
    #     (outputs_pth['conv1'].detach().numpy().swapaxes(1, 3).swapaxes(1, 2) - tff("sub/y:0")) /
    #     tff("truediv/y:0") * tff("mul/y:0") + tff("BiasAdd/bias:0") -
    #     tff("BiasAdd:0")
    # )
    print(
        outputs_pth['conv3'].detach().numpy().swapaxes(1, 3).swapaxes(1, 2) -
        tff("6-convolutional:0")
    )
    print(
        outputs_pth['conv8'].detach().numpy().swapaxes(1, 3).swapaxes(1, 2) -
        tff("20-convolutional:0")
    )
    print(
        outputs_pth['fc1'].detach().numpy() -
        tff("output:0")
    )
    print([[0.00112486, -0.00016582, -0.00093555, -0.00109863, -0.00133038, -0.00104356]])

    print()
