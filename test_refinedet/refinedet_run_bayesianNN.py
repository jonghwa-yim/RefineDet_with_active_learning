"""
In this example, we will load a RefineDet model and use it to detect objects.
"""

import argparse
import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt

import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))

HOMEDIR = os.path.expanduser("~") + '/'
CUR_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
# config_path = os.path.abspath(CUR_PATH+"/../config.cfg")


def get_final_output_MC_dropout(resP3_mcdrop, resP4_mcdrop, resP5_mcdrop, resP6_mcdrop):

    return None


def get_entropy_sum(y):
    tmp = np.exp(y - np.max(y, 0))
    softmaxed = tmp / tmp.sum(0)
    entropy_sum = (-1 * softmaxed * np.log(softmaxed)).sum()
    return entropy_sum


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found
    return labelnames


def ShowResults(img, target_file, results, labelmap, threshold=0.6, save_fig=False, show_fig=False):
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()

    num_classes = len(labelmap.item) - 1
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    for i in range(0, results.shape[0]):
        score = results[i, -2]
        if threshold and score < threshold:
            continue

        label = int(results[i, -1])
        name = get_labelname(labelmap, label)[0]
        color = colors[label % num_classes]

        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
        display_text = '%s: %.2f' % (name, score)
        ax.text(xmin, ymin, display_text, fontsize='small', bbox={'facecolor': color, 'alpha': 0.1})
    if save_fig:
        # plt.savefig(image_file[:-4] + '_dets.jpg', bbox_inches="tight")
        plt.savefig(target_file, bbox_inches="tight", dpi=300)
        print('Saved: ' + target_file)
    if show_fig:
        plt.show()
    return


def compute_BALD(net, transformer, nT, im_names, show_mcdropout_result=False):
    resP6_feat = np.zeros((nT, 120, 8, 8), np.float32)
    resP5_feat = np.zeros((nT, 120, 16, 16), np.float32)
    resP4_feat = np.zeros((nT, 120, 32, 32), np.float32)
    resP3_feat = np.zeros((nT, 120, 64, 64), np.float32)

    info_BALD_P6 = np.zeros(len(im_names), np.float32)
    info_BALD_P5 = np.zeros(len(im_names), np.float32)
    info_BALD_P4 = np.zeros(len(im_names), np.float32)
    info_BALD_P3 = np.zeros(len(im_names), np.float32)

    for i, im_name in enumerate(im_names):
        resP6_feat.fill(0.0)
        resP5_feat.fill(0.0)
        resP4_feat.fill(0.0)
        resP3_feat.fill(0.0)

        image_file = HOMEDIR + '/data/coco/images/val/' + im_name
        image = caffe.io.load_image(image_file)
        transformed_image = transformer.preprocess('data', image)

        for k in range(nT):
            net.blobs['data'].data[...] = transformed_image

            # detections = net.forward()['detection_out']
            net.forward()
            # det_label = detections[0, 0, :, 1]

            resP6_feat[k] = net.blobs['resP6_inter_mbox_conf'].data[0].copy()
            resP5_feat[k] = net.blobs['resP5_inter_mbox_conf'].data[0].copy()
            resP4_feat[k] = net.blobs['resP4_inter_mbox_conf'].data[0].copy()
            resP3_feat[k] = net.blobs['resP3_inter_mbox_conf'].data[0].copy()

            # run_test_model(resP6_feat[k])   # test code

        # ================= compute MC-dropout feature map =================
        resP6_mcdrop = resP6_feat.sum(0) / nT
        resP5_mcdrop = resP5_feat.sum(0) / nT
        resP4_mcdrop = resP4_feat.sum(0) / nT
        resP3_mcdrop = resP3_feat.sum(0) / nT
        # ==================================================================
        # ================= get final output of MC-dropout =================
        detections = get_final_output_MC_dropout(resP3_mcdrop, resP4_mcdrop, resP5_mcdrop, resP6_mcdrop)

        # ==================================================================
        # ========== get Entropy H[y|x, D) from each feature map ===========
        tmp = (resP6_mcdrop[0:40] + resP6_mcdrop[40:80] + resP6_mcdrop[80:120]) / 3
        resP6_entropy_sum = get_entropy_sum(tmp) / 64

        tmp = (resP5_mcdrop[0:40] + resP5_mcdrop[40:80] + resP5_mcdrop[80:120]) / 3
        resP5_entropy_sum = get_entropy_sum(tmp) / 256

        tmp = (resP4_mcdrop[0:40] + resP4_mcdrop[40:80] + resP4_mcdrop[80:120]) / 3
        resP4_entropy_sum = get_entropy_sum(tmp) / 1024

        tmp = (resP3_mcdrop[0:40] + resP3_mcdrop[40:80] + resP3_mcdrop[80:120]) / 3
        resP3_entropy_sum = get_entropy_sum(tmp) / 4096
        # ==================================================================

        # ======== get Expected Entropy E(H[y|x, w]) and BALD value ========
        expected_entropy = 0.0
        for k in range(nT):
            tmp = (resP6_feat[k][0:40] + resP6_feat[k][40:80] + resP6_feat[k][80:120]) / 3
            expected_entropy += get_entropy_sum(tmp) / 64
        info_BALD_P6[i] = resP6_entropy_sum - expected_entropy

        expected_entropy = 0.0
        for k in range(nT):
            tmp = (resP5_feat[k][0:40] + resP5_feat[k][40:80] + resP5_feat[k][80:120]) / 3
            expected_entropy += get_entropy_sum(tmp) / 64
        info_BALD_P5[i] = resP5_entropy_sum - expected_entropy

        expected_entropy = 0.0
        for k in range(nT):
            tmp = (resP4_feat[k][0:40] + resP4_feat[k][40:80] + resP4_feat[k][80:120]) / 3
            expected_entropy += get_entropy_sum(tmp) / 64
        info_BALD_P4[i] = resP4_entropy_sum - expected_entropy

        expected_entropy = 0.0
        for k in range(nT):
            tmp = (resP3_feat[k][0:40] + resP3_feat[k][40:80] + resP3_feat[k][80:120]) / 3
            expected_entropy += get_entropy_sum(tmp) / 64
        info_BALD_P3[i] = resP3_entropy_sum - expected_entropy
        # ==================================================================

        if show_mcdropout_result:
            show_MCDropout_result(transformed_image, resP6_mcdrop, resP5_mcdrop, resP4_mcdrop, resP3_mcdrop)

    info_BALD_sum = info_BALD_P6 + info_BALD_P5 + info_BALD_P4 + info_BALD_P3

    # sorted_BALD_P6_index = np.argsort(info_BALD_P6)
    # sorted_BALD_P5_index = np.argsort(info_BALD_P5)
    # sorted_BALD_P4_index = np.argsort(info_BALD_P4)
    # sorted_BALD_P3_index = np.argsort(info_BALD_P3)
    sorted_BALD_sum_idx = np.argsort(info_BALD_sum)

    return sorted_BALD_sum_idx


def show_MCDropout_result(transformed_image, resP6_mcdrop, resP5_mcdrop, resP4_mcdrop, resP3_mcdrop):
    # load MC-dropout model
    model_def_bayes = os.path.abspath(CUR_PATH + 'bayesian_dropout_model/deploy_multiple_input.prototxt')
    model_weights_bayes = os.path.abspath(CUR_PATH + '../models/ResNet/coco/refinedet_resnet101_512x512/coco_refinedet_resnet101_512x512_iter_52000.caffemodel')
    net_bayes = caffe.Net(model_def_bayes, model_weights_bayes, caffe.TEST)

    # image preprocessing
    if '320' in model_def_bayes:
        img_resize = 320
    else:
        img_resize = 512
    net_bayes.blobs['data'].reshape(1, 3, img_resize, img_resize)

    net_bayes.blobs['data'].data[...] = transformed_image
    net_bayes.blobs['dataP6'].data[...] = resP6_mcdrop
    net_bayes.blobs['dataP5'].data[...] = resP5_mcdrop
    net_bayes.blobs['dataP4'].data[...] = resP4_mcdrop
    net_bayes.blobs['dataP3'].data[...] = resP3_mcdrop

    res_bayes = net_bayes.forward()

    # load labelmap
    labelmap_file = 'data/coco/labelmap_coco.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    detections = res_bayes['detection_out']
    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3] * image.shape[1]
    det_ymin = detections[0, 0, :, 4] * image.shape[0]
    det_xmax = detections[0, 0, :, 5] * image.shape[1]
    det_ymax = detections[0, 0, :, 6] * image.shape[0]
    result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])

    ShowResults(image, None, result, labelmap, save_fig=False, show_fig=True)
    return


def get_pool_point_list(im_names):
    # load MC-dropout model
    model_def = os.path.abspath(CUR_PATH + 'bayesian_dropout_model/refinedet_resnet101_512x512_MCDrop_deploy.prototxt')
    model_weights = os.path.abspath(CUR_PATH + '../models/ResNet/coco/refinedet_resnet101_512x512/coco_refinedet_resnet101_512x512_iter_52000.caffemodel')
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # image preprocessing
    if '320' in model_def:
        img_resize = 320
    else:
        img_resize = 512
    net.blobs['data'].reshape(1, 3, img_resize, img_resize)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    nT = 8

    sorted_pool_point_idx = compute_BALD(net, transformer, nT, im_names)

    return sorted_pool_point_idx


def draw_results(im_names, sorted_pool_point_idx):
    # Get normal detection of images
    # load normal model
    model_def = os.path.abspath(CUR_PATH + '/../models/ResNet/coco/refinedet_resnet101_512x512/deploy.prototxt')
    model_weights = os.path.abspath(CUR_PATH + '/../models/ResNet/coco/refinedet_resnet101_512x512/coco_refinedet_resnet101_512x512_iter_52000.caffemodel')
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # load labelmap
    labelmap_file = HOMEDIR + 'data/coco/labelmap_coco.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    # image preprocessing
    if '320' in model_def:
        img_resize = 320
    else:
        img_resize = 512
    net.blobs['data'].reshape(1, 3, img_resize, img_resize)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    if not os.path.exists('results_bayesianNN'):
        os.mkdir('results_bayesianNN')
    target_folder = 'results_bayesianNN/info_BALD_sum'
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.mkdir(target_folder)

    for i, idx in enumerate(sorted_pool_point_idx):
        target_name = format(i, '06d') + '_' + im_names[idx]

        image_file = HOMEDIR + 'data/coco/images/val/' + im_names[idx]
        image = caffe.io.load_image(image_file)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        detections = net.forward()['detection_out']
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3] * image.shape[1]
        det_ymin = detections[0, 0, :, 4] * image.shape[0]
        det_xmax = detections[0, 0, :, 5] * image.shape[1]
        det_ymax = detections[0, 0, :, 6] * image.shape[0]
        result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])

        # show result
        ShowResults(image, os.path.join(target_folder, target_name), result, labelmap, 0.6, save_fig=True)

    return


def create_uncertain_image_list(im_names, sorted_pool_point_idx):
    fp = open(HOMEDIR + 'my_projects/RefineDet-master/results_bayesianNN/uncertain_image_list.txt', 'w')
    for i, idx in enumerate(sorted_pool_point_idx):
        img_id = im_names[idx][:-4]
        fp.write(img_id + '\n')
    fp.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_fig', action='store_true')
    args = parser.parse_args()

    # gpu preparation
    if args.gpu_id >= 0:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    im_names = os.listdir(HOMEDIR + 'data/coco/images/val/')
    im_names_partial = im_names[:100]   # Temporary code

    sorted_pool_point_idx = get_pool_point_list(im_names)

    create_uncertain_image_list(im_names, sorted_pool_point_idx)

    # draw_results(im_names_partial, sorted_pool_point_idx)

    print('Done')
