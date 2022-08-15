import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from utils import bounding_rectagle


def get_flip_func(prob_horizontal=0.5, prob_vertical=0.5):
    """
    Returns a function for randomly flipping an image and its ellipse annotations.

    :param prob_horizontal: float
        The probability of performing a horizontal flip.
    :param prob_vertical: float
        The probability of performing a vertical flip.
    :return: function
        A function that performs a random flip.
    """

    assert 0 <= prob_horizontal <= 1, \
        "Invalid parameter: prob_horizontal={}. Value must be in [0,1]".format(prob_horizontal)

    assert 0 <= prob_vertical <= 1, \
        "Invalid parameter: prob_vertical={}. Value must be in [0,1]".format(prob_vertical)

    @tf.function
    def random_flip(image, image_size, gt_annot, gt_labels):
        """
        Randomly flip an image and its annotations horizontally and/or vertically.

        :param image: tf.Tensor of shape (height, width, channels) OR tf.RaggedTensor of shape (None, None, channels)
            The input image tensor.
        :param image_size: tf.Tensor of shape (3,)
            Tensor with the height, width, and channels of the input image.
        :param gt_annot: tf.RaggedTensor of shape (None, 5)
            Tensor with the parameters of ellipse annotations
        :param gt_labels: tf.RaggedTensor
            Tensor with the annotation labels.
        :return: (tf.Tensor OR tf.RaggedTensor), tf.Tensor, tf.RaggedTensor, tf.RaggedTensor
            The flipped image, its shape, the flipped annotations, and their labels.
        """

        pi = tf.constant(np.pi)
        horizontal = tf.random.uniform((), 0, 1)
        vertical = tf.random.uniform((), 0, 1)

        ragged = False
        if type(image) == tf.RaggedTensor:
            image = image.to_tensor(shape=image_size)
            ragged = True

        height = image_size[0]
        width = image_size[1]
        height = tf.cast(height, dtype=tf.float32)
        width = tf.cast(width, dtype=tf.float32)

        gt_annot = gt_annot.to_tensor()

        if horizontal < prob_horizontal:
            image = tf.image.flip_left_right(image)
            xc = width - gt_annot[:, 0]
            yc = gt_annot[:, 1]
            rx = gt_annot[:, 2]
            ry = gt_annot[:, 3]
            theta = pi - gt_annot[:, 4]
            gt_annot = tf.stack((xc, yc, rx, ry, theta), axis=1)

        if vertical < prob_vertical:
            image = tf.image.flip_up_down(image)
            xc = gt_annot[:, 0]
            yc = height - gt_annot[:, 1]
            rx = gt_annot[:, 2]
            ry = gt_annot[:, 3]
            theta = -1.0 * gt_annot[:, 4]
            gt_annot = tf.stack((xc, yc, rx, ry, theta), axis=1)

        gt_annot = tf.RaggedTensor.from_tensor(gt_annot, row_splits_dtype=tf.int32)

        # If the input image was stored in a ragged tensor, then convert the result into a ragged tensor.
        if ragged:
            image = tf.RaggedTensor.from_tensor(image, row_splits_dtype=tf.int32)

        return image, image_size, gt_annot, gt_labels

    return random_flip


def get_change_image_func(max_brightness_delta=0.2,
                          lower_contrast_factor=0.5,
                          upper_contrast_factor=2.0,
                          lower_saturation_factor=0.75,
                          upper_saturation_factor=1.25,
                          max_hue_delta=0.1):
    """
    Returns a function for randomly changing the image brightness, contrast, saturation, and hue.

    :param max_brightness_delta: float
        A delta parameter randomly picked in the [-max_brightness_delta, max_brightness_delta) interval will be
        passed to the tf.image.adjust_brightness() method. Must be non-negative.
    :param lower_contrast_factor: float
        A contrast factor randomly picked in the [lower_contrast_factor, upper_contrast_factor) will be
        passed to the tf.image.adjust_contrast() method.
    :param upper_contrast_factor: float
        A contrast factor randomly picked in the [lower_contrast_factor, upper_contrast_factor) will be
        passed to the tf.image.adjust_contrast() method.
    :param lower_saturation_factor: float
        A saturation factor randomly picked in the [lower_saturation_factor, upper_saturation_factor) will be
        passed to the tf.image.adjust_saturation() method.
    :param upper_saturation_factor: float
        A saturation factor randomly picked in the [lower_saturation_factor, upper_saturation_factor) will be
        passed to the tf.image.adjust_saturation() method.
    :param max_hue_delta: float
        A delta parameter randomly picked in the [-max_hue_delta, max_hue_delta) interval will be
        passed to the tf.image.adjust_hue() method. Must be non-negative.
    :return: function
        A function that performs a random in the image attributes.
    """

    assert max_brightness_delta >= 0,\
        "Invalid parameter: max_brightness_delta={}. Value must be non-negative".format(max_brightness_delta)

    @tf.function
    def random_change_image(image, image_size, gt_annot, gt_labels):
        """
        Randomly modify the brightness, contrast, saturation and hue of an image.

        :param image: tf.Tensor of shape (height, width, 3) OR tf.RaggedTensor of shape (None, None, 3)
            The input image tensor.
        :param image_size: tf.Tensor of shape (3,)
            Tensor with the height, width, and channels of the input image.
        :param gt_annot: tf.RaggedTensor
            Tensor with the annotation parameters.
        :param gt_labels: tf.RaggedTensor
            Tensor with the annotation labels.
        :return: (tf.Tensor OR tf.RaggedTensor), tf.Tensor, tf.RaggedTensor, tf.RaggedTensor
            The modified image, its shape, annotations, and labels.
        """

        ragged = False
        if type(image) == tf.RaggedTensor:
            image = image.to_tensor(shape=image_size)
            ragged = True

        image = tf.image.random_brightness(image, max_brightness_delta)
        image = tf.image.random_contrast(image, lower_contrast_factor, upper_contrast_factor)
        image = tf.image.random_saturation(image, lower_saturation_factor, upper_saturation_factor)
        image = tf.image.random_hue(image, max_hue_delta)

        # If the input image was stored in a ragged tensor, then convert the result into a ragged tensor.
        if ragged:
            image = tf.RaggedTensor.from_tensor(image, row_splits_dtype=tf.int32)

        return image, image_size, gt_annot, gt_labels

    return random_change_image


def get_rotate_func(prob_rot=0.5,
                    max_rot_angle=tf.constant(np.pi / 4.),
                    crop=True):
    """
    Returns a function for randomly rotating an image and its ellipse annotations.

    :param prob_rot: float
        The probability of rotating the image, must be in the [0,1] interval. Default: 0.5.
    :param max_rot_angle: float
        The maximal rotation angle in radians. The rotation angle will be randomly picked in the
        [-max_rot_angle, max_rot_angle) interval. Default: pi/4.
    :param crop: bool
        Whether to crop the image and its annotations to the largest inner axis-aligned rectangle
        without padding.
    :return: function
        A function that performs a random rotation on an image and its annotations.
    """

    assert 0 <= prob_rot <= 1,\
        "Invalid parameter: prob_rot={}. Value must be in the [0,1] interval.".format(prob_rot)

    @tf.function
    def random_rotate(image, image_size, gt_annot, gt_labels):
        """
        Randomly rotate an image and its annotations.

        :param image: tf.Tensor of shape (height, width, channels) OR tf.RaggedTensor of shape (None, None, channels)
            The input image tensor.
        :param image_size: tf.Tensor of shape (3,)
            Tensor with the height, width, and channels of the input image.
        :param gt_annot: tf.RaggedTensor of shape (None, 5)
            Tensor with the parameters of ellipse annotations
        :param gt_labels: tf.RaggedTensor
            Tensor with the annotation labels.
        :return: (tf.Tensor OR tf.RaggedTensor), tf.Tensor, tf.RaggedTensor, tf.RaggedTensor
            The rotated (an optionally cropped) image, its shape, the flipped annotations, and their labels.
        """

        if prob_rot <= tf.random.uniform((), 0, 1):
            return image, image_size, gt_annot, gt_labels

        height = image_size[0]
        width = image_size[1]

        ragged = False
        if type(image) == tf.RaggedTensor:
            image = image.to_tensor(shape=image_size)
            ragged = True

        gt_annot = gt_annot.to_tensor()
        gt_labels = gt_labels.to_tensor()

        # get random rotation angle:
        angle = tf.random.uniform((), -max_rot_angle, max_rot_angle)
        # angle = -np.pi/45

        # rotate image, canvas will be of the same height and width, empty areas will be filled
        image = tfa.image.rotate(image, angle, interpolation='bilinear', fill_mode='constant', fill_value=0.0)

        # compute the parameters of the rotated ellipses:
        img_center_x = tf.cast(tf.round(width / 2), dtype=tf.float32)
        img_center_y = tf.cast(tf.round(height / 2), dtype=tf.float32)
        gt_annot = rotate_ellipse(gt_annot, angle, img_center_x, img_center_y)

        # crop the image to the maximum possible size without filling pixels
        if crop:
            # compute shape of cropped image:
            target_size = inner_rect(width, height, angle)
            # apply crop:
            offset_width = tf.round(width / 2) - image_size[1] / 2
            offset_height = tf.round(height / 2) - image_size[0] / 2
            # tf.print(offset_height, offset_width, image_size[0], image_size[1])
            image, gt_annot, gt_labels = \
                crop_image_with_ellipses(image, gt_annot, gt_labels,
                                         offset_height, offset_width, target_size[0], target_size[1])

        image_size = tf.shape(image)

        if ragged:
            image = tf.RaggedTensor.from_tensor(image, row_splits_dtype=tf.int32)

        gt_annot = tf.RaggedTensor.from_tensor(gt_annot, row_splits_dtype=tf.int32)
        gt_labels = tf.RaggedTensor.from_tensor(gt_labels, row_splits_dtype=tf.int32)

        return image, image_size, gt_annot, gt_labels

    return random_rotate


def get_crop_func(target_height=640,
                  target_width=640):
    """
    Returns a function for randomly cropping an image and its ellipse annotations to a user-specified size.

    :param target_height: int
        The height of the image after cropping.
    :param target_width: int
        The width of the image after cropping.
    :return: function
        A function that performs a random crop of the input image to a given size.
    """

    assert type(target_height) == int and target_height > 0, \
        "Invalid parameter: target_height={}. Value must be a non-negative integer.".format(target_height)

    assert type(target_width) == int and target_width > 0, \
        "Invalid parameter: target_height={}. Value must be a non-negative integer.".format(target_width)

    @tf.function(experimental_relax_shapes=True)
    def random_crop_image(image, image_size, gt_annot, gt_labels):
        """
        Randomly rotate an image and its annotations.

        :param image: tf.Tensor of shape (height, width, channels) OR tf.RaggedTensor of shape (None, None, channels)
            The input image tensor.
        :param image_size: tf.Tensor of shape (3,)
            Tensor with the height, width, and channels of the input image.
        :param gt_annot: tf.RaggedTensor of shape (None, 5)
            Tensor with the parameters of ellipse annotations
        :param gt_labels: tf.RaggedTensor
            Tensor with the annotation labels.
        :return: (tf.Tensor OR tf.RaggedTensor), tf.Tensor, tf.RaggedTensor, tf.RaggedTensor
            The rotated (and optionally cropped) image, its shape, the flipped annotations, and their labels.
        """

        ragged = False
        if type(image) == tf.RaggedTensor:
            image = image.to_tensor(shape=image_size)
            ragged = True

        orig_height = image_size[0]
        orig_width = image_size[1]
        gt_annot = gt_annot.to_tensor()
        gt_labels = gt_labels.to_tensor()

        if orig_height - target_height > 0:
            offset_height = tf.random.uniform((), 0, orig_height - target_height, dtype=tf.int32)
        else:
            offset_height = 0
        if orig_width - target_width > 0:
            offset_width = tf.random.uniform((), 0, orig_width - target_width, dtype=tf.int32)
        else:
            offset_width = 0

        image, gt_annot, gt_labels = \
            crop_image_with_ellipses(image, gt_annot, gt_labels,
                                     offset_height, offset_width, target_height, target_width)

        gt_annot = tf.RaggedTensor.from_tensor(gt_annot, row_splits_dtype=tf.int32)
        gt_labels = tf.RaggedTensor.from_tensor(gt_labels, row_splits_dtype=tf.int32)

        image_size = tf.shape(image)

        if ragged:
            image = tf.RaggedTensor.from_tensor(image, row_splits_dtype=tf.int32)

        return image, image_size, gt_annot, gt_labels

    return random_crop_image


@tf.function
def crop_image_with_ellipses(image,
                             gt_annot,
                             gt_labels,
                             offset_height,
                             offset_width,
                             target_height,
                             target_width):
    """
    Crop an image and its annotations to a given size with a given offset.
    Annotations with centers falling outside the cropped image will be omitted.

    :param image: tf.Tensor
        An image tensot of shape (height, width, channels).
    :param gt_annot: tf.Tensor
        A Tensor with ellipse annotation parameters of shape (num_objects, 5).
    :param gt_labels: tf.Tensor
        A Tensor with annotation labels of shape (num_objects, ) OR (num_objects, num_classes)
    :param offset_height: int
        Vertical offset of the cropped image from the top.
    :param offset_width: int
        Horizontal offset of the cropped image from the left.
    :param target_height: int
        Height of the image after cropping.
    :param target_width: int
        Width of the image after cropping.
    :return: tf.Tensor, tf.Tensor, tf.Tensor
            The cropped image, its annotations and labels.
    """

    offset_height = tf.cast(offset_height, dtype=tf.int32)
    offset_width = tf.cast(offset_width, dtype=tf.int32)
    target_height = tf.cast(target_height, dtype=tf.int32)
    target_width = tf.cast(target_width, dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, target_height, target_width
    )

    xc = gt_annot[:, 0] - tf.cast(offset_width, dtype=tf.float32)
    yc = gt_annot[:, 1] - tf.cast(offset_height, dtype=tf.float32)
    rx = gt_annot[:, 2]
    ry = gt_annot[:, 3]
    theta = gt_annot[:, 4]
    gt_annot = tf.stack((xc, yc, rx, ry, theta), axis=1)

    xc = tf.cast(xc, dtype=tf.int32)
    yc = tf.cast(yc, dtype=tf.int32)

    xmask = tf.logical_and(tf.greater(xc, 0), tf.less(xc, target_width))
    ymask = tf.logical_and(tf.greater(yc, 0), tf.less(yc, target_height))
    xymask = tf.logical_and(xmask, ymask)

    gt_annot = tf.boolean_mask(gt_annot, xymask, axis=0)
    gt_labels = tf.boolean_mask(gt_labels, xymask, axis=0)

    return image, gt_annot, gt_labels


def get_preprocess_func(model):
    """
    Returns a function for preprocessing an image with the method of an object detection model.

    :param model: object
        An object detection model instance implementing a `preprocess()` method.
    :return: function
        A function that performs preprocessing of an image.
    """
    def preprocess(image, image_shape, gt_annot, gt_labels):
        """
        Preprocess an image with an object detection models `preprocess()` method.

        :param image: tf.Tensor of shape (height, width, channels) OR tf.RaggedTensor of shape (None, None, channels)
            The input image tensor.
        :param image_shape: tf.Tensor of shape (3,)
            Tensor with the height, width, and channels of the input image.
        :param gt_annot: tf.RaggedTensor of shape (None, 5)
            Tensor with the parameters of ellipse annotations
        :param gt_labels: tf.RaggedTensor
            Tensor with the annotation labels.
        :return: tf.Tensor, tf.Tensor, tf.RaggedTensor, tf.RaggedTensor
            The preprocessed image, its shape, annotations, and labels.
        """

        if type(image) == tf.RaggedTensor:
            image = image.to_tensor(shape=image_shape)

        image = tf.expand_dims(image, axis=0)
        image = tf.squeeze(model.preprocess(image)[0])

        image_shape = tf.shape(image)

        return image, image_shape, gt_annot, gt_labels

    return preprocess


# @tf.function
def ell2box(image, image_shape, gt_ellipses, gt_labels):
    """
    Converts annotation ellipses of an image to bounding boxes
    by computing the bounding rectangle for each ellipse.

    :param image: tf.Tensor of shape (height, width, channels) OR tf.RaggedTensor of shape (None, None, channels)
        The input image tensor.
    :param image_shape: tf.Tensor of shape (3,)
        Tensor with the height, width, and channels of the input image.
    :param gt_ellipses: tf.RaggedTensor of shape (None, 5)
        Tensor with the parameters of ellipse annotations.
    :param gt_labels: tf.RaggedTensor
        Tensor with the annotation labels.
    :return: (tf.RaggedTensor OR tf.Tensor), tf.Tensor, tf.RaggedTensor, tf.RaggedTensor
        The preprocessed image, its shape, bounding boxes, and labels.
    """

    gt_ellipses = gt_ellipses.to_tensor()

    gt_boxes = bounding_rectagle(gt_ellipses)

    height = tf.cast(image.shape[0], dtype=tf.float32)
    width = tf.cast(image.shape[1], dtype=tf.float32)

    # ymin, xmin, ymax, xmax

    ymin = gt_boxes[:, 0] / height
    ymax = gt_boxes[:, 2] / height

    xmin = gt_boxes[:, 1] / width
    xmax = gt_boxes[:, 3] / width

    gt_boxes = tf.stack((ymin, xmin, ymax, xmax), axis=1)

    gt_boxes = tf.maximum(gt_boxes, 0.0)
    gt_boxes = tf.minimum(gt_boxes, 1.0)

    gt_boxes = tf.RaggedTensor.from_tensor(gt_boxes, row_splits_dtype=tf.int32)

    return image, image_shape, gt_boxes, gt_labels


def rotate_ellipse(ellipse_params,
                   angle,
                   img_center_x,
                   img_center_y):
    """
    Rotate ellipse annotations in an image that is rotated around its center.

    :param ellipse_params: tf.Tensor
        A Tensor of shape (n_objects, 5) with the ellipse parameters for each object:
        (x_center, y_center, x_radius, y_radius, rot. angle)
    :param angle: float
        The angle of rotation in radians.
    :param img_center_x: int
        The x center coordinate of the image.
    :param img_center_y: int
        The y center coordinate of the image.
    :return: tf.Tensor
        Tensor of shape (n_objects, 5) with the parameters of the rotated ellipses.
    """

    ellipse_centers = ellipse_params[:, 0:2]
    ellipse_size_x = ellipse_params[:, 2]
    ellipse_size_y = ellipse_params[:, 3]
    ellipse_angles = ellipse_params[:, 4]

    # Rotation matrix:
    cos = tf.cos(angle)
    sin = tf.sin(angle)
    m00 = cos
    m01 = sin
    m10 = -sin
    m11 = cos
    rot_matrix = tf.convert_to_tensor([[m00, m01], [m10, m11]])
    # tf.print('rot. matrix: ' + str(rot_matrix))

    # compute the bounding dimensions of the rotated image
    # new_width = (height * tf.abs(sin)) + (width * tf.abs(cos))
    # new_height = (height * tf.abs(cos)) + (width * tf.abs(sin))
    # adjust the rotation matrix to take into account translation
    # translation_x = (1 - cos) * cx - sin * cy + (new_width / 2.) - cx
    # translation_y = sin * cx + (1 - cos) * cy + (new_height / 2.) - cy
    translation_x = (1 - cos) * img_center_x - sin * img_center_y
    translation_y = sin * img_center_x + (1 - cos) * img_center_y

    rotated_centers = tf.transpose(tf.matmul(rot_matrix, ellipse_centers, transpose_b=True))

    rotated_centers_x = rotated_centers[:, 0] + translation_x
    rotated_centers_y = rotated_centers[:, 1] + translation_y

    rotated_ellipse_angles = ellipse_angles - angle
    rotated_ellipse_params = tf.stack((rotated_centers_x, rotated_centers_y,
                                       ellipse_size_x, ellipse_size_y, rotated_ellipse_angles), axis=1)

    return rotated_ellipse_params


def inner_rect(width, height, angle):
    """
    Compute the size of the largest possible axis-aligned inner rectangle of an image of size (width x height),
    that has been rotated by 'angle' radians.

    :param width: int
        The width of the original, rotated image.
    :param height: int
        The height of the original, rotated image.
    :param angle: float
        The rotation angle in radians.
    :return: tf.Tensor
        A tensor with two elements: the height and width of the inner rectangle.
    """
    w = tf.cast(width, dtype=tf.float32)
    h = tf.cast(height, dtype=tf.float32)

    if w <= 0 or h <= 0:
        return tf.stack((0, 0))

    # width_is_longer = w >= h
    # side_long, side_short = (w, h) if width_is_longer else (h, w)
    if w >= h:
        side_long = w
        side_short = h
    else:
        side_long = h
        side_short = w

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a = tf.abs(tf.sin(angle))
    cos_a = tf.abs(tf.cos(angle))

    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        # wr, hr = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)

        if w >= h:
            wr = x / sin_a
            hr = x / cos_a
        else:
            wr = x / cos_a
            hr = x / sin_a

    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return tf.cast(tf.stack((hr, wr)), dtype=tf.int32)


def random_rotate_box(image, gt_boxes, gt_labels):
    max_rot_angle = tf.constant(np.pi / 20.)

    angle = tf.random.uniform((), -max_rot_angle, max_rot_angle)
    height = image.shape[0]
    width = image.shape[1]
    cx = width // 2
    cy = height // 2
    image = tfa.image.rotate(image, angle, interpolation='bilinear', fill_mode='constant', fill_value=0.0)

    # compute corners:
    gt_boxes = gt_boxes.to_tensor()
    corners = get_corners(gt_boxes, width, height)
    rotated_corners = rotate_boxes(corners, angle, cx, cy, height, width)
    # tf.print('corners.shape = ' + str(corners.shape))
    # tf.print(corners)
    # tf.print('rotated_corners.shape = ' + str(rotated_corners.shape))
    # tf.print(rotated_corners)

    new_gt_boxes = get_enclosing_box(rotated_corners, height, width)
    # tf.print(new_gt_boxes.shape)
    # tf.print('-----------------------')

    new_gt_boxes = tf.RaggedTensor.from_tensor(new_gt_boxes, row_splits_dtype=tf.int32)

    return image, new_gt_boxes, gt_labels


def get_enclosing_box(corners, height, width):
    x_coords = tf.stack((corners[:, 0], corners[:, 2], corners[:, 4], corners[:, 6]), axis=1)
    y_coords = tf.stack((corners[:, 1], corners[:, 3], corners[:, 5], corners[:, 7]), axis=1)
    # y_coords = corners[:, [1, 3, 5, 7]]

    # tf.print(x_coords.shape)

    xmin = tf.reduce_min(x_coords, axis=1) / width
    ymin = tf.reduce_min(y_coords, axis=1) / height
    xmax = tf.reduce_max(x_coords, axis=1) / width
    ymax = tf.reduce_max(y_coords, axis=1) / height

    bboxes = tf.stack((ymin, xmin, ymax, xmax), axis=1)

    return bboxes


def rotate_boxes(corners, angle, cx, cy, height, width):
    corners = tf.reshape(corners, shape=(-1, 2))

    cx = tf.cast(cx, dtype=tf.float32)
    cy = tf.cast(cy, dtype=tf.float32)
    # width = tf.cast(width, dtype=tf.float32)
    # height = tf.cast(height, dtype=tf.float32)

    # Rotation matrix:
    cos = tf.cos(angle)
    sin = tf.sin(angle)
    m00 = cos
    m01 = sin
    m10 = -sin
    m11 = cos
    rot_matrix = tf.convert_to_tensor([[m00, m01], [m10, m11]])
    # tf.print('rot. matrix: ' + str(rot_matrix))

    # compute the bounding dimensions of the rotated image
    new_width = (height * tf.abs(sin)) + (width * tf.abs(cos))
    new_height = (height * tf.abs(cos)) + (width * tf.abs(sin))
    # adjust the rotation matrix to take into account translation
    translation_x = (1 - cos) * cx - sin * cy + (new_width / 2.) - cx
    translation_y = sin * cx + (1 - cos) * cy + (new_height / 2.) - cy

    # rot_matrix = tf.convert_to_tensor([[m00, m01, m02], [m10, m11, m12]])

    rotated_corners = tf.transpose(tf.matmul(rot_matrix, corners, transpose_b=True))

    rotated_corners_x = rotated_corners[:, 0] + translation_x
    rotated_corners_y = rotated_corners[:, 1] + translation_y

    rotated_corners = tf.stack((rotated_corners_x, rotated_corners_y), axis=1)

    rotated_corners = tf.reshape(rotated_corners, shape=(-1, 8))

    return rotated_corners


def get_corners(gt_boxes, im_width, im_height):
    widths = gt_boxes[:, 3] - gt_boxes[:, 1]
    heights = gt_boxes[:, 2] - gt_boxes[:, 0]
    y1 = gt_boxes[:, 0] * im_height
    x1 = gt_boxes[:, 1] * im_width
    y2 = y1
    x2 = x1 + widths * im_width
    y3 = y1 + heights * im_height
    x3 = x1
    y4 = gt_boxes[:, 2] * im_height
    x4 = gt_boxes[:, 3] * im_width

    corners = tf.stack((x1, y1, x2, y2, x3, y3, x4, y4), axis=1)

    return corners
