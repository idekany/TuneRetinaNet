import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import json
from six import BytesIO
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Ellipse, Rectangle
from object_detection.utils import visualization_utils as viz_utils


def load_json_images_annotations(filename: str, label_dict: dict, subdir: str = "."):
    """
    Extract the list of images, ground truth annotations, and their corresponding classes from a
    single json annotation file.
    NOTE: this function expects json files created with the VIA (VGG Image Annotator) tool.

    :param filename: str
        The name of the input list file
    :param label_dict: dict
        A dictionary with the class names and integer classes (starting with 1) as key-value pairs,
        e.g.: {'apple': 1, 'orange': 2, 'pear': 3}
    :param subdir: str
        The name of the subdirectory containing the images referred to in the annotation file.
    :return: (list of str, list of np.ndarray, list of np.ndarray)
        The lists of images, annotations, and labels.
    """

    with open(filename) as f:
        annotations = json.load(f)

    image_list = []
    box_list = []
    label_list = []

    for ii, entry in enumerate(annotations.values()):

        image_file = entry['filename']
        image_file = os.path.join(subdir, image_file)
        # print(image_file)
        image_list.append(image_file)
        # get image size:
        with Image.open(image_file) as im:
            img_width, img_height = im.size

        boxes = []
        labels = []
        for region in entry['regions']:
            assert region['shape_attributes']['name'] == 'rect'
            # read rectangle parameters:
            x = region['shape_attributes']['x']
            y = region['shape_attributes']['y']
            width = region['shape_attributes']['width']
            height = region['shape_attributes']['height']
            # read class label:
            label = region['region_attributes']['class']

            xmin = x / img_width
            ymin = y / img_height
            xmax = (x + width) / img_width
            ymax = (y + height) / img_height

            boxes.append([ymin, xmin, ymax, xmax])
            labels.append(label_dict[label])

        box_list.append(np.array(boxes, dtype=np.float32))
        label_list.append(np.array(labels, dtype=np.int32))

    return image_list, box_list, label_list


def load_json_images_annotations_from_list(filename: str, label_dict: dict, subdir: str = ".",
                                           annot_suffix: str = ".json"):
    """
    Extract the list of images, ground truth annotations, and their corresponding classes from a
    list of image files and corresponding json annotation files.
    NOTE: this function expects json files created with the VIA (VGG Image Annotator) tool.

    :param filename: str
        The name of the input list file
    :param label_dict: dict
        A dictionary with the class names and integer classes (starting with 1) as key-value pairs,
        e.g.: {'apple': 1, 'orange': 2, 'pear': 3}
    :param subdir: str
        The name of the subdirectory containing the images and the corresponding annotation files.
    :param annot_suffix: str
        The suffix of the annotation files such as: image_file='example.jpeg' -> annotation_file='example<annot_suffix>'
    :return: (list of str, list of np.ndarray, list of np.ndarray)
        The lists of images, annotations, and labels.
    """

    with open(filename) as f:
        files = np.loadtxt(f, dtype='str')

    image_list = []
    annot_list = []
    label_list = []

    for ii, image_file in enumerate(files):

        image_list.append(os.path.join(subdir, image_file))

        annot_file = image_file.strip('.jpg').strip('.jpeg').strip('.png') + annot_suffix
        with open(os.path.join(subdir, annot_file)) as af:
            annotations = json.load(af)

        # skip uppermost dict level with only one entry:
        annotations = list(annotations.values())[0]

        assert annotations['filename'] == image_file, "Image filename `{}` differs from annotation file attribute `{}`" \
            .format(image_file, annotations['filename'])

        annot = []
        labels = []
        for region in annotations['regions']:

            assert region['shape_attributes']['name'] == 'rect' or region['shape_attributes']['name'] == 'ellipse'

            if region['shape_attributes']['name'] == 'rect':
                # read rectangle parameters:
                x = region['shape_attributes']['x']
                y = region['shape_attributes']['y']
                w = region['shape_attributes']['width']
                h = region['shape_attributes']['height']
                rx = w / 2.
                ry = h / 2.
                cx = x + rx
                cy = y + ry
                theta = 0.
                # read class label:
                label = region['region_attributes']['class']

                # overwrite ellipse entry to rectangle entry in json:
                region['shape_attributes'] = \
                    {'name': 'ellipse', 'cx': round(cx), 'cy': round(cy), 'rx': rx, 'ry': ry, 'theta': theta}

            else:
                # read ellipse parameters:
                cx = region['shape_attributes']['cx']
                cy = region['shape_attributes']['cy']
                rx = region['shape_attributes']['rx']
                ry = region['shape_attributes']['ry']
                theta = region['shape_attributes']['theta']
                # read class label:
                label = region['region_attributes']['class']

            annot.append([cx, cy, rx, ry, theta])
            labels.append(label_dict[label])

        annot_list.append(np.array(annot, dtype=np.float32))
        label_list.append(np.array(labels, dtype=np.int32))

    return image_list, annot_list, label_list


def load_image_into_numpy_array(filename):
    """
    Load an image from file into a numpy array of shape
    (height, width, channels), where channels=3 for RGB.

    :param filename: str
        Path to th input file.
    :return: numpy.ndarray, uint8
        The array with the input image.
    """
    img_data = tf.io.gfile.GFile(filename, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def plot_detections(image_np, boxes, classes, scores, category_index, min_score_thresh=0.8, image_name=None):
    """
    Wrapper function for the object_detection.utils.visualization_utils.visualize_boxes_and_labels_on_image_array()
    method.

    :param image_np: numpy.ndarray, uint8
        Array with the input image with shape (height, width, 3).
    :param boxes: numpy.ndarray
        Array with the bounding box parameters of shape (n_objects, 4).
    :param classes: numpy.ndarray
        Array with the class labels of shape (n_objects, ).
        Indices must be 1-based, and must match the keys in `category_index`.
    :param scores: numpy.ndarray
        Array with the detection scores. If None, groundtruth boxes are assumed,
        and all boxes will be plotted as black with neither classes nor scores.
    :param category_index: dict
        Dictionary of category dictionaries (each holding a category index `id` and category name `name`)
        keyed by category indices.
    :param min_score_thresh: float
        The minimum required score for a box to be shown.
    :param image_name: str
        Name of the output image file.
    """

    image_np_annotated = image_np.copy()
    image_np_annotated = viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_annotated, boxes, classes, scores, category_index,
        use_normalized_coordinates=True, min_score_thresh=min_score_thresh)
    if image_name is not None:
        plt.imsave(image_name, image_np_annotated)
    else:
        plt.imshow(image_np_annotated)


def plot_images_with_boxes(images, gt_boxes, gt_labels,
                           pred_boxes, pred_labels, pred_scores,
                           category_index, label_id_offset,
                           figname='image_list_with_boxes', figformat='jpg',
                           min_score_thresh=0.5,
                           max_boxes_to_draw=20,
                           skip_scores=False, skip_labels=False):
    """
    Plot a list / batch of images with the ground truth and
    (optionally) predicted boxes (with labels and scores) overlaid.

    :param images: array-like with tf.Tensor elements of shape (height, width, channels)
    OR tf.Tensor with batch dimension of shape (batch_size, height, width, channels)
        The list or batch of image tensors to be plotted.
    :param gt_boxes: array-like with tf.Tensor elements of shape (n_boxes, 4)
    OR tf.Tensor with batch dimension of shape (batch_size, n_boxes, 4)
        The list or batch of ground truth boxes to be plotted.
    :param gt_labels: array-like with tf.Tensor elements of shape (n_boxes, n_classes) or (n_boxes,)
    OR tf.Tensor with batch dimension of shape (batch_size, n_boxes, n_classes) or (batch_size, n_boxes)
        The list or batch of ground truth labels.
    :param pred_boxes: array-like with tf.Tensor elements of shape (n_boxes, 4)
    OR tf.Tensor with batch dimension of shape (batch_size, n_boxes, 4)
    OR None
        The list or batch of predicted boxes to be plotted. If None, only the ground truth boxes will be plotted.
    :param pred_labels: array-like with tf.Tensor elements of shape (n_boxes, n_classes) or (n_boxes,)
    OR tf.Tensor with batch dimension of shape (batch_size, n_boxes, n_classes) or (batch_size, n_boxes)
    OR None
        The list or batch of predicted labels. If None, only the ground truth boxes will be plotted.
    :param pred_scores: array-like with tf.Tensor elements of shape (n_boxes,)
    OR tf.Tensor with batch dimension of shape (batch_size, n_boxes)
    OR None
        The list or batch of prediction scores. If None, only the ground truth boxes will be plotted.
    :param category_index: dict
        A dictionary containing category dictionaries (each holding category index `id` and category name `name`)
        keyed by category indices.
    :param label_id_offset: int
        The offset of label id's with respect to a labelling scheme that starts with 0.
    :param figname: str
        The path and name of the output figure.
    :param figformat: str
        File format of the output figure. Only valid pyplot output formats are allowed.
    :param min_score_thresh: float
        The minimum detection score threshold for a predicted object to be plotted.
    :param max_boxes_to_draw: int OR None
        The maximum number of detection boxes to be plotted. If None, draw all boxes.
    :param skip_scores: boolean
        Whether to skip the drawing of bounding boxes.
    :param skip_labels: boolean
        Whether to skip score when drawing a single detection.
    :return:
    """

    n_img = len(images)
    image_shape = tf.shape(images[0]).numpy()
    scaler = MinMaxScaler(feature_range=(0, 255))

    ncols = 3
    nrows = int(np.ceil(n_img / ncols))
    if nrows == 1:
        ncols = n_img
    fig = plt.figure(figsize=(ncols * 10, 10 * nrows))
    for ii in range(n_img):

        plt.subplot(nrows, ncols, ii + 1)

        image_np = scaler.fit_transform(images[ii].numpy().reshape(-1, 1)). \
            reshape(image_shape).astype('int32')

        gt_boxes_np = gt_boxes[ii].numpy()

        # check if ground truth labels are one-hot encoded:
        if tf.shape(gt_labels[0]).numpy().shape[0] > 1:
            gt_labels_np = tf.argmax(gt_labels[ii], axis=1).numpy().flatten().astype('int32')
        else:
            gt_labels_np = gt_labels[ii].numpy().astype('int32')

        image_np_annotated = image_np.copy()
        image_np_annotated = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_annotated, gt_boxes_np, gt_labels_np + label_id_offset, None, category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=max_boxes_to_draw, groundtruth_box_visualization_color='black', line_thickness=1)

        if None not in (pred_boxes, pred_labels, pred_scores):

            pred_boxes_np = pred_boxes[ii].numpy()

            # check if predicted labels are one-hot encoded:
            if tf.shape(pred_labels[0]).numpy().shape[0] > 1:
                pred_labels_np = tf.argmax(pred_labels[ii], axis=1).numpy().flatten().astype('int32')
            else:
                pred_labels_np = pred_labels[ii].numpy().astype('int32')

            pred_scores_np = pred_scores[ii].numpy()

            image_np_annotated = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_annotated, pred_boxes_np, pred_labels_np + label_id_offset, pred_scores_np, category_index,
                use_normalized_coordinates=True, min_score_thresh=min_score_thresh,
                max_boxes_to_draw=max_boxes_to_draw, line_thickness=1,
                skip_scores=skip_scores, skip_labels=skip_labels)

        plt.imshow(image_np_annotated)

    plt.tight_layout()
    plt.savefig(figname + '.' + figformat, format=figformat)
    fig.clf()
    plt.close(fig)
    del fig


def plot_image_batch_with_boxes(dataset,
                                category_index,
                                label_id_offset,
                                rescale: bool = True,
                                figname: str = 'image_batch_with_boxes',
                                figformat: str = 'jpg'):
    """
    Plots a batch of images with their corresponding object bounding boxes and labels.

    :param dataset: tf.Dataset
        A batched tensorflow dataset object containing the entries:
        image, image shape, boxes, labels
    :param category_index: dict
        A dictionary containing category dictionaries (each holding category index `id` and category name `name`)
        keyed by category indices.
    :param label_id_offset: int
        The offset of label id's with respect to a labelling scheme that starts with 0.
    :param rescale: bool
        Whether to rescale the image into the [0, 255] range.
    :param figname: str
        The filename for the output figure.
    :param figformat: str
        The format of the output figure. Valid matplotlib.pyplot formats are accepted.
    """

    image_list_np = []
    boxes_list = []
    labels_list = []

    if rescale:
        scaler = MinMaxScaler(feature_range=(0, 255))
    else:
        scaler = None

    for img, img_shape, boxes, labels in dataset.unbatch():

        if scaler is not None:
            image_list_np.append(scaler.fit_transform(img.numpy().reshape(-1, 1)).reshape(img.shape).astype('int32'))
        else:
            image_list_np.append(img.numpy())
        boxes_list.append(boxes.numpy())
        labels_list.append(tf.argmax(labels.to_tensor(), axis=1).numpy().flatten().astype('int32'))

    n_img = len(image_list_np)

    ncols = 3
    nrows = int(np.ceil(n_img / ncols))
    fig = plt.figure(figsize=(30, 10 * nrows))

    for ii in range(n_img):

        plt.subplot(nrows, ncols, ii + 1)

        image_np_annotated = image_list_np[ii].copy()

        image_np_annotated = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_annotated,
            boxes_list[ii],
            labels_list[ii] + label_id_offset,
            np.ones([boxes_list[ii].shape[0]]),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=None,
            groundtruth_box_visualization_color='black',
            skip_scores=True,
            skip_labels=False,
            line_thickness=2)

        plt.imshow(image_np_annotated)

    plt.tight_layout()
    plt.savefig(figname + '.' + figformat, format=figformat)
    fig.clf()
    plt.close(fig)
    del fig


def plot_image_batch_with_ellipses(data_batch, category_index, ell2box=False, rescale=True,
                                   figname='image_batch_with_ellipses', figformat='pdf'):
    """
    Plots a batch of images with their corresponding object annotation ellipses and labels.

    :param data_batch: tf.Dataset
        A batched tensorflow dataset object containing the entries:
        image, image shape, ellipses, labels
    :param category_index: dict
        A dictionary containing category dictionaries (each holding category index `id` and category name `name`)
        keyed by category indices.
    :param ell2box: bool
        Whether to also plot the bounding rectangles of the ellipses.
    :param rescale: bool
        Whether to rescale the image into the [0, 255] range.
    :param figname: str
        The filename for the output figure.
    :param figformat: str
        The format of the output figure. Valid matplotlib.pyplot formats are accepted.
    """
    image_list_np = []
    shapes_list = []
    ellipse_list = []
    box_list = []
    labels_list = []

    if rescale:
        scaler = MinMaxScaler(feature_range=(0, 255))
    else:
        scaler = None

    for img, img_shape, ellipses, labels in data_batch.unbatch():
        # image_list_np.append(img.numpy().astype('int32'))
        if scaler is not None:
            image_list_np.append(scaler.fit_transform(img.numpy().reshape(-1, 1)).reshape(img.shape).astype('int32'))
        else:
            image_list_np.append(img.numpy())
        # print(img_shape)
        shapes_list.append(img_shape.numpy().astype('int32'))
        ellipse_list.append(ellipses.numpy())
        labels_list.append(labels.numpy().astype('int32'))
        if ell2box:
            ellipses = ellipses.to_tensor()
            boxes = bounding_rectagle(ellipses)
            box_list.append(boxes.numpy())

    for ii, image_np in enumerate(image_list_np):

        dpi = 400

        fig = plt.figure(figsize=(20, 10), dpi=dpi, tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        _ = ax.imshow(image_np, aspect=1, interpolation='none')

        plt.rcParams["figure.autolayout"] = True

        # plt.subplot(nrows, ncols, ii + 1)
        plot_gt_ellipses(ax, ellipse_list[ii], labels_list[ii], category_index, np.ones_like(ellipse_list[ii]))
        if box_list:
            plot_gt_boxes(ax, box_list[ii], labels_list[ii], category_index, np.ones_like(ellipse_list[ii]))

        cy = int(image_np.shape[0] // 2)
        cx = int(image_np.shape[1] // 2)

        # print(shapes_list[ii], cy, cx)
        plot_bb(ax, cx, cy, shapes_list[ii])
        # print(annot_list[ii])
        # print(labels_list[ii])

        if figname is not None:
            plt.savefig(figname + '_' + str(ii + 1) + '.' + figformat, format=figformat, dpi=dpi)
        else:
            plt.show()


def plot_bb(axis, cx, cy, shape):
    height, width, channels = shape

    rr = Rectangle(xy=(cx - width / 2, cy - height / 2), width=width, height=height)
    axis.add_artist(rr)
    rr.set_clip_box(axis.bbox)
    rr.set_color('green')
    rr.set_alpha(1)
    rr.set_linewidth(4)
    rr.set_fill(False)


def bounding_rectagle(ellipse_parameters):
    """
    Compute the parameters of the bounding rectangles for a set of ellipses.

    :param ellipse_parameters: tf.Tensor
        Tensor of shape (n_objects, 5) with the ellipse parameters:
        x_center, y_center, x_radius, y_radius, rotation angle.
    :return: tf.Tensor
        Tensor of shape (n_objects, 4) with the resulting bounding box parameters:
        ymin, xmin, ymax, xmax
    """
    cx = ellipse_parameters[:, 0]
    cy = ellipse_parameters[:, 1]
    rx = ellipse_parameters[:, 2]
    ry = ellipse_parameters[:, 3]
    theta = ellipse_parameters[:, 4]

    pi = tf.constant(np.pi)
    epsilon = 1e-10

    tx1 = tf.atan(-1 * (ry * tf.sin(theta)) / (rx * tf.cos(theta) + epsilon))
    tx2 = tx1 + pi
    # print(tx1, tx2)

    x1 = rx * tf.cos(theta) * tf.cos(tx1) - ry * tf.sin(theta) * tf.sin(tx1)
    x2 = rx * tf.cos(theta) * tf.cos(tx2) - ry * tf.sin(theta) * tf.sin(tx2)
    # print(x1, x2)

    # ty1 = np.arctan((ry * tf.cos(theta)) / (rx * tf.sin(theta) + epsilon))
    ty1 = tf.atan((ry * tf.cos(theta)) / (rx * tf.sin(theta) + epsilon))
    ty2 = ty1 + pi
    # print(ty1, ty2)

    y1 = rx * tf.sin(theta) * tf.cos(ty1) + ry * tf.cos(theta) * tf.sin(ty1)
    y2 = rx * tf.sin(theta) * tf.cos(ty2) + ry * tf.cos(theta) * tf.sin(ty2)
    # print(y1, y2)

    half_width = tf.reduce_max(tf.stack((x1, x2), axis=0), axis=0)
    half_height = tf.reduce_max(tf.stack((y1, y2), axis=0), axis=0)
    # tf.print(half_width)
    # tf.print(half_width.shape)

    ymin = cy - half_height
    xmin = cx - half_width
    ymax = cy + half_height
    xmax = cx + half_width

    rectangle_params = tf.stack((ymin, xmin, ymax, xmax), axis=1)

    return rectangle_params


def plot_gt_ellipses(axis, ellipses, classes, category_index, scores):

    for i, annot in enumerate(ellipses):
        cx, cy, rx, ry, theta = annot

        ell = Ellipse(xy=(cx, cy), width=2 * rx, height=2 * ry, angle=theta * 180.0 / np.pi, zorder=i + 2)
        axis.add_artist(ell)
        ell.set_clip_box(axis.bbox)
        ell.set_color('black')
        ell.set_alpha(1)
        ell.set_linewidth(0.2)
        ell.set_fill(False)


def plot_gt_boxes(axis, boxes, classes, category_index, scores):

    for i, annot in enumerate(boxes):
        ymin, xmin, ymax, xmax = annot

        rr = Rectangle(xy=(xmin, ymin), width=xmax - xmin, height=ymax - ymin)
        axis.add_artist(rr)
        rr.set_clip_box(axis.bbox)
        rr.set_color('blue')
        rr.set_alpha(1)
        rr.set_linewidth(0.2)
        rr.set_fill(False)


def get_lists_from_batch(data_batch):
    # Unpack the ragged tensors of this batch.
    # The first dimension of each ragged tensor is the batch size.
    images_batch_rtensor, gt_boxes_rtensor, gt_classes_rtensor = data_batch

    # Convert the ragged tensors of this batch to lists of tensors:
    images_list = tf.split(images_batch_rtensor, images_batch_rtensor.shape[0], axis=0)
    images_list = [tf.squeeze(item.to_tensor(), axis=0) for item in images_list]

    gt_boxes_list = tf.split(gt_boxes_rtensor, gt_boxes_rtensor.shape[0], axis=0)
    gt_boxes_list = [tf.squeeze(item.to_tensor(), axis=0) for item in gt_boxes_list]

    gt_classes_list = tf.split(gt_classes_rtensor, gt_classes_rtensor.shape[0], axis=0)
    gt_classes_list = [tf.squeeze(item.to_tensor(), axis=0) for item in gt_classes_list]

    return images_list, gt_boxes_list, gt_classes_list


def get_list_from_ragged_batch(data_batch):
    # Convert the ragged tensors of this batch to lists of tensors:
    tensor_list = tf.split(data_batch, data_batch.shape[0], axis=0)
    tensor_list = [tf.squeeze(item.to_tensor(), axis=0) for item in tensor_list]

    return tensor_list


def get_list_from_batch(data_batch):
    # Convert the ragged tensors of this batch to lists of tensors:
    tensor_list = tf.split(data_batch, data_batch.shape[0], axis=0)
    tensor_list = [tf.squeeze(item, axis=0) for item in tensor_list]

    return tensor_list


@tf.function()
def compute_iou_matrix(box_arr1, box_arr2):
    """
    Compute the IOU matrix for two sets of bounding boxes.

    :param box_arr1: tf.Tensor
        Tensor of shape (n_objects, 4) with the first set of bounding box parameters.
    :param box_arr2: tf.Tensor
        Tensor of shape (n_objects, 4) with the second set of bounding box parameters.
    :return: tf.Tensor
        The resulting IOU matrix.
    """

    epsilon = tf.constant(1e-9, dtype='float32')

    x11, y11, x12, y12 = tf.split(box_arr1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(box_arr2, 4, axis=1)

    xA = tf.maximum(x11, tf.transpose(x21))
    yA = tf.maximum(y11, tf.transpose(y21))
    xB = tf.minimum(x12, tf.transpose(x22))
    yB = tf.minimum(y12, tf.transpose(y22))

    interArea = tf.maximum((xB - xA + epsilon), 0) * tf.maximum((yB - yA + epsilon), 0)
    boxAArea = (x12 - x11 + epsilon) * (y12 - y11 + epsilon)
    boxBArea = (x22 - x21 + epsilon) * (y22 - y21 + epsilon)

    iou_matrix = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)

    return iou_matrix


@tf.function()
def compute_map_iou_per_image(gt_boxes_tensor, gt_labels_tensor, pred_boxes_tensor, pred_labels_tensor,
                              pred_boxes_scores,
                              iou_threshold=0.5, n_scores=100):
    """
    Compute the mean average precision (mAP) and the mean IOU for an image. The mean is taken across all classes,
    and in case of the mean IOU, across all score thresholds for each class.
    :param gt_boxes_tensor: tf.Tensor, shape=(n_boxes,4), dtype=float32
        A tensor with the ground truth boxes holding (ymin, xmin, ymax, xmax) values for each box in
        relative coordinates in [0,1].
    :param gt_labels_tensor: tf.Tensor, shape=(n_boxes,), dtype=int32
        A tensor with the ground truth class labels (starting from 0).
    :param pred_boxes_tensor: tf.Tensor, shape=(n_boxes,4), dtype=float32
        A tensor with the predicted boxes holding (ymin, xmin, ymax, xmax) values for each box in
        relative coordinates in [0,1].
    :param pred_labels_tensor: tf.Tensor, shape=(n_boxes,), dtype=int32
        A tensor with the predicted class labels (starting from 0).
    :param pred_boxes_scores: tf.Tensor, shape=(n_boxes,), dtype=float32
        A tensor with the probability scores (of the predicted class) of the predicted boxes.
    :param iou_threshold: float
        Threshold of the IOU metric in the computation of the mean average precision (mAP).
    :param n_scores: int
        The number of score thresholds for sampling the precision-recall curve.
    :return: (tf.Tensor, tf.Tensor)
        The mean average precision (across classes) and the mean IOU (across scores and classes).
    """
    epsilon = tf.constant(1e-10)

    classes = tf.unique(gt_labels_tensor).y  # determine the unique classes present in current image, ...
    num_cl = tf.shape(classes)[0]  # ... and count them

    # initialize tensor array for aggregating each average precision value per class
    average_precisions = tf.TensorArray(tf.float32, size=num_cl, dynamic_size=False, clear_after_read=True)
    scores_maxf1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
    mean_ious = tf.TensorArray(tf.float32, size=num_cl, dynamic_size=False, clear_after_read=True)

    # loop over the classes present in the current image:
    for jj in tf.range(num_cl):

        i_class = classes[jj]

        # initialize tensor arrays for aggregating the precisions and recalls for each score threshold:
        precisions = tf.TensorArray(tf.float32, size=n_scores, dynamic_size=False, clear_after_read=True)
        recalls = tf.TensorArray(tf.float32, size=n_scores, dynamic_size=False, clear_after_read=True)
        ious = tf.TensorArray(tf.float32, size=n_scores, dynamic_size=False, clear_after_read=True)
        f1scores = tf.TensorArray(tf.float32, size=n_scores, dynamic_size=False, clear_after_read=True)

        # get the ground truth boxes corresponding to the current (i_class) class:
        index_gt_class = \
            tf.squeeze(tf.where(
                tf.equal(gt_labels_tensor, i_class)
            ), axis=1)
        gt_boxes_tensor_class = tf.reshape(tf.gather(gt_boxes_tensor, index_gt_class), shape=(-1, 4))

        # get the scores corresponding to the current (i_class) class:
        pred_boxes_scores_class = tf.gather(pred_boxes_scores,
                                            tf.squeeze(tf.where(tf.equal(pred_labels_tensor, i_class))))
        # determine max score for current class:
        max_score_class = tf.reduce_max(pred_boxes_scores_class)
        # create score grid for current class for sampling the precision-recall curve:
        scores = tf.cast(tf.linspace(0.0, max_score_class, n_scores), dtype='float32')
        # NOTE: the number of true positives for a score threshold above the maximum score will be zero,
        # therefore the recall will be undefined. For the cases, the precision vs recall curve takes the constant
        # value of precision=1 at all recalls by definition. We account for this by setting the upper limit for
        # the score grid to max(score) for the class, and by adding the last precision recall point at the end of
        # the loop below:

        # for i_score, score in enumerate(scores[:-1]):
        for i_score in tf.range(n_scores - 1):
            score = scores[i_score]

            # get the predicted boxes corresponding to the current (i_class) class:
            index_pred_class = \
                tf.squeeze(tf.where(
                    tf.logical_and(tf.equal(pred_labels_tensor, i_class),
                                   tf.greater_equal(pred_boxes_scores, score))
                ), axis=1)
            pred_boxes_tensor_class = tf.gather(pred_boxes_tensor, index_pred_class)

            # Compute IOU matrix: rows correspond to gt boxes, columns to predicted boxes of current class:
            iou_matrix_class = compute_iou_matrix(gt_boxes_tensor_class, pred_boxes_tensor_class)

            mean_iou_boxes = tf.reduce_mean(tf.reduce_max(iou_matrix_class, axis=1))

            # Compute the number of true positives for this class:
            # count the rows in `iou_matrix_class` that have (at least) one iou > iou_threshold column
            tp = tf.reduce_sum(tf.cast(
                tf.reduce_any(tf.greater_equal(iou_matrix_class, iou_threshold), axis=1),
                dtype='float32'))

            # Compute the number of false negatives for this class:
            # count the rows in `iou_matrix_class` that do not have any iou > iou_threshold column
            fn = tf.reduce_sum(tf.cast(
                tf.logical_not(tf.reduce_any(tf.greater_equal(iou_matrix_class, iou_threshold), axis=1)),
                dtype='float32'))

            # Compute the number of false positives for this class:
            # count the columns in `iou_matrix_class` that do not have any iou > iou_threshold row
            fp1 = tf.reduce_sum(tf.cast(
                tf.logical_not(tf.reduce_any(tf.greater_equal(iou_matrix_class, iou_threshold), axis=0)),
                dtype='float32'))
            # for each row in `iou_matrix_class`, count all redundant iou > iou_threshold columns
            #     get a boolean mask for the rows with at least one detection
            mask = tf.reduce_any(tf.greater_equal(iou_matrix_class, iou_threshold), axis=1)
            #     get a subset of the iou matrix with the above boolean mask
            iou_matrix_class_detections = tf.boolean_mask(iou_matrix_class, mask)
            #     count all redundant detections
            fp2 = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.greater_equal(iou_matrix_class_detections, iou_threshold),
                                                      dtype='float32'), axis=1) - 1)

            fp = fp1 + fp2

            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)

            f1score = 2 * precision * recall / (precision + recall + epsilon)

            recalls = recalls.write(i_score, recall)
            precisions = precisions.write(i_score, precision)
            f1scores = f1scores.write(i_score, f1score)
            ious = ious.write(i_score, mean_iou_boxes)

        recalls = recalls.write(n_scores - 1, 0)
        precisions = precisions.write(n_scores - 1, 1)
        f1scores = f1scores.write(n_scores - 1, 0)

        recalls = recalls.stack()
        precisions = precisions.stack()
        mean_iou_class = tf.reduce_mean(ious.stack())
        mean_ious = mean_ious.write(jj, mean_iou_class)

        # compute AP without interpolation:
        average_precision = tf.abs(-tf.reduce_sum(tf.experimental.numpy.diff(recalls) * precisions[:-1]))
        average_precisions = average_precisions.write(jj, average_precision)

        # compute detection score at maximum f1score:
        f1scores = f1scores.stack()
        argmax_f1scores = tf.argmax(f1scores)
        score_maxf1 = tf.gather(scores, argmax_f1scores)
        # save detection score at maximum f1score for each class only if it is non-zero:
        if tf.greater(score_maxf1, 0.):
            scores_maxf1 = scores_maxf1.write(jj, tf.stack([tf.cast(i_class, dtype=tf.float32), score_maxf1], axis=0))

    mean_iou = tf.reduce_mean(mean_ious.stack())

    # Stack the score_maxf1 values. The returned tensor will have two columns, the first will hold the classes
    # for which scores_maxf1 is non-zero, and the second will hold the scores_maxf1 value for that class.
    scores_maxf1 = scores_maxf1.stack()

    mean_average_precision = tf.reduce_mean(average_precisions.stack())

    return mean_average_precision, mean_iou, scores_maxf1


@tf.function()
def compute_map_iou_per_batch(gt_boxes_tensors, gt_one_hot_labels_tensors, detections, batch_size,
                              iou_threshold=0.5, n_scores=100, num_classes=1):
    # initialize tensor array for aggregating mAP values for each image in the batch:
    mean_average_precisions_batch = \
        tf.TensorArray(tf.float32, size=batch_size, dynamic_size=False, clear_after_read=True)
    mean_iou_batch = \
        tf.TensorArray(tf.float32, size=batch_size, dynamic_size=False, clear_after_read=True)
    scores_maxf1_batch = \
        tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True, infer_shape=False)
    scores_maxf1 = \
        tf.TensorArray(tf.float32, size=num_classes, dynamic_size=False, clear_after_read=True)

    for ii in range(batch_size):
        # invert one-hot encodings back to 'dense' labels:
        gt_labels_tensor = tf.cast(tf.argmax(gt_one_hot_labels_tensors[ii], axis=1), dtype='int32')
        gt_boxes_tensor = gt_boxes_tensors[ii]

        # unpack the predicted classes and boxes from the `detections` dictionary:
        pred_labels_tensor = tf.cast(detections['detection_classes'][ii], dtype='int32')
        pred_boxes_tensor = detections['detection_boxes'][ii]
        pred_boxes_scores = detections['detection_scores'][ii]

        # ------------------------------------------------------------------------------

        mean_average_precision, mean_iou, scores_maxf1_img = \
            compute_map_iou_per_image(gt_boxes_tensor, gt_labels_tensor,
                                      pred_boxes_tensor, pred_labels_tensor, pred_boxes_scores,
                                      iou_threshold=iou_threshold, n_scores=n_scores)

        mean_average_precisions_batch = mean_average_precisions_batch.write(ii, mean_average_precision)
        mean_iou_batch = mean_iou_batch.write(ii, mean_iou)
        scores_maxf1_batch = scores_maxf1_batch.write(ii, scores_maxf1_img)

    map_batch = tf.reduce_mean(mean_average_precisions_batch.stack())
    mean_iou_batch = tf.reduce_mean(mean_iou_batch.stack())

    scores_maxf1_all = scores_maxf1_batch.concat()
    # Loop through the classes and determine the mean score_maxf1. If there were no detections for that class
    # in this batch, the score_maxf1 or it will be nan.
    for i_class in tf.range(num_classes):
        # mask = tf.equal(scores_maxf1_all[:, 0], tf.cast(i_class, tf.float32))
        # if tf.not_equal(tf.size(mask), 0):
        # scores_maxf1_class = tf.boolean_mask(scores_maxf1_all[:, 1], mask, axis=0)

        index_class = tf.where(tf.equal(scores_maxf1_all[:, 0], tf.cast(i_class, tf.float32)))
        scores_maxf1_class = tf.gather(scores_maxf1_all[:, 1], index_class)
        scores_maxf1 = scores_maxf1.write(i_class, tf.reduce_mean(scores_maxf1_class))

    scores_maxf1 = scores_maxf1.stack()

    return map_batch, mean_iou_batch, scores_maxf1


def get_datagen(image_path_list, gt_boxes_list, gt_labels_list, num_classes, label_id_offset):
    """
    Returns a data generator for feeding a tensorflow.Dataset object.

    :param image_path_list: array-like
        List of the image files.
    :param gt_boxes_list: list of numpy.ndarray
        List of bounding box arrays corresponding to the images in `image_path_list`.
        Each array has a shape of (n_boxes, 4) where the 4 columns contain the (ymin, xmin, ymax, xmax) values in
        relative coordinates in [1,0].
    :param gt_labels_list: list of numpy.ndarray
        List of classification label arrays.
    :param num_classes: int
        The total number of ground truth classes
    :param label_id_offset: int
        The offset of label id's with respect to a labelling scheme that starts with 0.
    :return: generator function
    """

    def datagen():
        for (image_path, gt_boxes_np, gt_labels_np) in zip(image_path_list, gt_boxes_list, gt_labels_list):
            # # Load next image into PIL format:
            # image_pil = tf.keras.utils.load_img(image_path)
            # # Convert the image into a numpy array:
            # image_np = tf.keras.preprocessing.image.img_to_array(image_pil, dtype='uint8')
            # # Covert the image array into tensor and add a batch dimension:
            # image_tensor = tf.expand_dims(tf.convert_to_tensor(image_np, dtype=tf.float32), axis=0)

            image = tf.io.read_file(image_path)
            image_tensor = tf.io.decode_image(image, channels=3, dtype=tf.uint8)
            image_shape = tf.convert_to_tensor(image_tensor.shape, dtype=tf.int32)
            image_tensor = tf.cast(image_tensor, dtype=tf.float32)

            # Run the image tensor through the model's preprocessing method
            #   this requires a batch dimension:
            # image_tensor = tf.expand_dims(image_tensor, axis=0)
            # image_tensor = tf.squeeze(model.preprocess(image_tensor)[0], axis=0)
            image_tensor = tf.RaggedTensor.from_tensor(image_tensor, row_splits_dtype=tf.int32)

            # Convert the groundtruth boxes from numpy array into tensor:
            gt_boxes_tensor = tf.convert_to_tensor(gt_boxes_np, dtype=tf.float32)
            gt_boxes_rtensor = tf.RaggedTensor.from_tensor(gt_boxes_tensor, row_splits_dtype=tf.int32)

            # Offset the groundtruth labels to start from 0,
            # convert the labels numpy array into tensor,
            # and change the labels into one-hot representation:
            zero_indexed_groundtruth_classes = tf.convert_to_tensor(gt_labels_np - label_id_offset)
            val_gt_one_hot_labels_tensor = tf.one_hot(zero_indexed_groundtruth_classes, num_classes)
            val_gt_one_hot_labels_rtensor = tf.RaggedTensor.from_tensor(val_gt_one_hot_labels_tensor,
                                                                        row_splits_dtype=tf.int32)

            yield image_tensor, image_shape, gt_boxes_rtensor, val_gt_one_hot_labels_rtensor

    return datagen
