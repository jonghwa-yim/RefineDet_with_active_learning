import os
import subprocess
import sys
import shutil
from random import shuffle
import json
from caffe.proto import caffe_pb2
from google.protobuf import text_format

HOMEDIR = os.path.expanduser("~")
PRJ_DIR = os.path.join(HOMEDIR, 'my_projects/RefineDet-master/')
COCO_DIR = os.path.join(PRJ_DIR, 'coco/PythonAPI/scripts/')
# The root directory which stores the coco images, annotations, etc.
DATA_DIR = "{}/data/coco/".format(HOMEDIR)

sys.path.append(PRJ_DIR)


class BASE_MODE:
    random = 0
    uncertainty_BALD = 1

COCO_NAMEs = [
    {
        'json_file': "instances_train2017-random",
        'train_list': "train-random.txt",
        'train_lmdb_out': "lmdb/train-random_lmdb",
        'train_data': "examples/coco/train-random_lmdb",
        'job_name': "refinedet_resnet101_{}_RD",
        'train_net_file': "{}/train-random.prototxt"
    }, {
        'json_file': "instances_train2017-dropout",
        'train_list': "train-dropout.txt",
        'train_lmdb_out': "lmdb/train-dropout_lmdb",
        'train_data': "examples/coco/train-dropout_lmdb",
        'job_name': "refinedet_resnet101_{}_AL",
        'train_net_file': "{}/train-dropout.prototxt"
    }, {
        'json_file': "instances_train2017",
        'train_list': "train.txt",
        'train_lmdb_out': "lmdb/train_lmdb",
        'train_data': "examples/coco/train_lmdb",
        'job_name': "refinedet_resnet101_{}",
        'train_net_file': "{}/train-random.prototxt"
    }
]


def move_val_points_to_train(im_names, pooling_points, json_name, redo=False):
    annotation_path = "{}/annotations/".format(DATA_DIR)
    out_json_file = annotation_path + json_name + '.json'

    if redo or not os.path.exists(out_json_file):

        with open(annotation_path + "instances_val2017_original.json", "r") as f:
            val = json.load(f)
        with open(annotation_path + "instances_train2017_original.json", "r") as f:
            train = json.load(f)

        pooling_points.sort()

        img_head = 0
        anno_head = 0
        for img_id in pooling_points:
            img_id = int(img_id)

            while img_head < len(val['images']):
                if val['images'][img_head]['id'] == img_id:
                    train["images"].append(val['images'][img_head])

                    ison = False
                    for anno_idx in range(anno_head, len(val["annotations"]), 1):
                        if val["annotations"][anno_idx]["image_id"] == img_id:
                            train["annotations"].append(val["annotations"][anno_idx])
                            ison = True
                        if ison and (val["annotations"][anno_idx]["image_id"] != img_id):
                            break
                    anno_head = anno_idx
                    img_head += 1
                    break
                img_head += 1

        with open(out_json_file, "w") as f:
            json.dump(train, f)

    return


def read_pool_point_from_random_list(num_batch_pooling):
    fp = open(PRJ_DIR + 'results_bayesianNN/random_image_list.txt', 'r')
    pooling_points = []
    for i in range(num_batch_pooling):
        buf = fp.readline().strip('\n')
        if len(buf) == 0:
            break
        pooling_points.append(buf)
    return pooling_points


def read_pool_point_from_file(num_batch_pooling):
    fp = open(PRJ_DIR + 'results_bayesianNN/uncertain_image_list.txt', 'r')
    pooling_points = []
    for i in range(num_batch_pooling):
        buf = fp.readline().strip('\n')
        if len(buf) == 0:
            break
        pooling_points.append(buf)
    return pooling_points


def batch_split_annotation(anno_sets, redo=False):
    print("batch_split_annotation running")
    ### Modify the address and parameters accordingly ###
    # The directory which contains the full annotation files for each set.
    anno_dir = "{}/annotations".format(DATA_DIR)
    # The root directory which stores the annotation for each image for each set.
    out_anno_dir = "{}/Annotations".format(DATA_DIR)
    # The directory which stores the imageset information for each set.
    imgset_dir = "{}/images".format(DATA_DIR)

    ### Process each set ###
    for i in range(0, len(anno_sets)):
        anno_set = anno_sets[i]
        anno_file = "{}/{}.json".format(anno_dir, anno_set)
        if not os.path.exists(anno_file):
            print("{} does not exist".format(anno_file))
            continue
        anno_name = anno_set.split("_")[-1]
        out_dir = "{}/{}".format(out_anno_dir, anno_name)
        imgset_file = "{}/{}.txt".format(imgset_dir, anno_name)
        if redo or not os.path.exists(out_dir):
            from coco.PythonAPI.scripts import split_annotation
            split_annotation.run_split_annotation(out_dir, imgset_file, anno_file, redo=False)
            # cmd = "python {}/split_annotation.py --out-dir={} --imgset-file={} {}" \
            #     .format(COCO_DIR, out_dir, imgset_file, anno_file)
            # print(cmd)
            # process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            # output = process.communicate()[0]
            # print(output)
    return


def batch_get_image_size(anno_sets, redo=False):
    print("batch_get_image_size running")
    ### Modify the address and parameters accordingly ###
    # The directory which contains the full annotation files for each set.
    anno_dir = "{}/annotations".format(DATA_DIR)
    # The directory which stores the imageset information for each set.
    imgset_dir = "{}/images".format(DATA_DIR)
    # The directory which stores the image id and size info.
    out_dir = "{}/data/coco".format(PRJ_DIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ### Get image size info ###
    for i in range(0, len(anno_sets)):
        anno_set = anno_sets[i]
        anno_file = "{}/{}.json".format(anno_dir, anno_set)
        if not os.path.exists(anno_file):
            continue
        anno_name = anno_set.split("_")[-1]
        imgset_file = "{}/{}.txt".format(imgset_dir, anno_name)
        if not os.path.exists(imgset_file):
            print("{} does not exist".format(imgset_file))
            sys.exit()
        name_size_file = "{}/{}_name_size.txt".format(out_dir, anno_name)
        if redo or not os.path.exists(out_dir):
            cmd = "python {}/get_image_size.py {} {} {}" \
                .format(COCO_DIR, anno_file, imgset_file, name_size_file)
            print(cmd)
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output = process.communicate()[0]
            print(output)
    return


def create_list_coco(train_list_file, anno_name, redo=False):
    # The directory name which holds the image sets.
    imgset_dir = "images"
    # The direcotry which contains the images.
    img_dir = "images"
    img_ext = "jpg"
    # The directory which contains the annotations.
    anno_dir = "Annotations"
    anno_ext = "json"
    subset = anno_name.split("_")[-1]

    if redo or os.path.exists(train_list_file):
        # Create training set.
        # We follow Ross Girschick's split.
        datasets = [subset]
        # subset = "train2017-dropout"
        img_files = []
        anno_files = []
        for dataset in datasets:
            imgset_file = "{}/{}/{}.txt".format(DATA_DIR, imgset_dir, dataset)
            with open(imgset_file, "r") as f:
                for line in f.readlines():
                    name = line.strip("\n")
                    # subset = name.split("_")[1]

                    isfound = False
                    img_file = "{}/{}/{}.{}".format(img_dir, 'train', name, img_ext)
                    if os.path.exists("{}/{}".format(DATA_DIR, img_file)):
                        isfound = True
                    else:
                        img_file = "{}/{}/{}.{}".format(img_dir, 'val', name, img_ext)
                        if os.path.exists("{}/{}".format(DATA_DIR, img_file)):
                            isfound = True
                    assert isfound, "{}/{} does not exist".format(DATA_DIR, img_file)
                    anno_file = "{}/{}/{}.{}".format(anno_dir, subset, name, anno_ext)
                    assert os.path.exists("{}/{}".format(DATA_DIR, anno_file)), \
                        "{}/{} does not exist".format(DATA_DIR, anno_file)
                    img_files.append(img_file)
                    anno_files.append(anno_file)
        # Shuffle the images.
        idx = [i for i in range(len(img_files))]
        shuffle(idx)
        with open(train_list_file, "w") as f:
            for i in idx:
                f.write("{} {}\n".format(img_files[i], anno_files[i]))
    return


def create_annoset_lmdb(list_file, out_dir, redo=False):
    global DATA_DIR
    example_dir = PRJ_DIR + "examples/coco"

    anno_type = "detection"
    label_type = "json"
    backend = "lmdb"
    check_size = False
    encode_type = "jpg"
    encoded = True
    gray = False
    label_map_file = PRJ_DIR + "data/coco/labelmap_coco.prototxt"
    min_dim = 0
    max_dim = 0
    resize_height = 0
    resize_width = 0
    shuffle = False
    check_label = True
    # ========================================= #

    # check if root directory exists
    if not os.path.exists(DATA_DIR):
        print("root directory: {} does not exist".format(DATA_DIR))
        sys.exit()
    # add "/" to root directory if needed
    if DATA_DIR[-1] != "/":
        DATA_DIR += "/"
    # check if list file exists
    if not os.path.exists(list_file):
        print("list file: {} does not exist".format(list_file))
        sys.exit()
    # check list file format is correct
    with open(list_file, "r") as lf:
        for line in lf.readlines():
            img_file, anno = line.strip("\n").split(" ")
            if not os.path.exists(DATA_DIR + img_file):
                print("image file: {} does not exist".format(DATA_DIR + img_file))

            if not os.path.exists(DATA_DIR + anno):
                print("annofation file: {} does not exist".format(DATA_DIR + anno))
                sys.exit()
            break
    # check if label map file exist
    if not os.path.exists(label_map_file):
        print("label map file: {} does not exist".format(label_map_file))
        sys.exit()
    label_map = caffe_pb2.LabelMap()
    lmf = open(label_map_file, "r")
    try:
        text_format.Merge(str(lmf.read()), label_map)
    except:
        print("Cannot parse label map file: {}".format(label_map_file))
        sys.exit()

    out_parent_dir = os.path.dirname(out_dir)
    if not os.path.exists(out_parent_dir):
        os.makedirs(out_parent_dir)
    if os.path.exists(out_dir) and not redo:
        print("{} already exists and I do not hear redo".format(out_dir))
        return
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    cmd = "{}/build/tools/convert_annoset" \
          " --anno_type={}" \
          " --label_type={}" \
          " --label_map_file={}" \
          " --check_label={}" \
          " --min_dim={}" \
          " --max_dim={}" \
          " --resize_height={}" \
          " --resize_width={}" \
          " --backend={}" \
          " --shuffle={}" \
          " --check_size={}" \
          " --encode_type={}" \
          " --encoded={}" \
          " --gray={}" \
          " {} {} {}" \
        .format(PRJ_DIR, anno_type, label_type, label_map_file, check_label,
                min_dim, max_dim, resize_height, resize_width, backend, shuffle,
                check_size, encode_type, encoded, gray, DATA_DIR, list_file, out_dir)
    print(cmd)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]

    if not os.path.exists(example_dir):
        os.makedirs(example_dir)
    link_dir = os.path.join(example_dir, os.path.basename(out_dir))
    if os.path.exists(link_dir):
        os.unlink(link_dir)
    os.symlink(out_dir, link_dir)

    print("Done")
    return


if __name__ == '__main__':
    from examples.refinedet.ResNet101_COCO_320_fn import run_COCO_detector_training
    from test_refinedet.refinedet_test import run_refinedet_test

    base_mode = BASE_MODE.random
    num_batch_pooling = 60
    total_val_data_num = 1368
    # ====== file names for active learning ======
    al_json_file = COCO_NAMEs[base_mode]['json_file']
    al_train_list_file = "{}data/coco/{}".format(PRJ_DIR, COCO_NAMEs[base_mode]['train_list'])
    al_train_lmdb_out = DATA_DIR + COCO_NAMEs[base_mode]['train_lmdb_out']
    # ============================================
    redo = True

    im_names = os.listdir(DATA_DIR + 'images/val/')
    read_pool_point_fn = read_pool_point_from_random_list if base_mode == 0 else read_pool_point_from_file

    for idx in range(1, int(total_val_data_num / num_batch_pooling) + 1):
        # sorted_pool_point_idx = refinedet_run_bayesianNN.get_pool_point_list(im_names_partial)
        pooling_points = read_pool_point_fn(num_batch_pooling * idx)

        move_val_points_to_train(im_names, pooling_points, al_json_file, redo=redo)
        anno_sets = [al_json_file]  # , "instances_val2017"]
        batch_split_annotation(anno_sets, redo=True)
        batch_get_image_size(anno_sets, redo=True)

        create_list_coco(al_train_list_file, al_json_file, redo=redo)
        create_annoset_lmdb(al_train_list_file, al_train_lmdb_out, redo=redo)

        run_COCO_detector_training(base_mode)
        run_refinedet_test(str(idx), prj_path=PRJ_DIR, base_mode=base_mode)

    print('Done')
