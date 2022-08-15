import sys
from tqdm import tqdm, trange
from utils import *
import imaug as ia
from callbacks import EarlyStopping, PlotLearning
from models import RetinaNetModel
from argparser import argparser, default_parameter_file


def main():
    # Read parameters from a file or from the command line:
    parser = argparser()
    # print(len(sys.argv))
    if len(sys.argv) == 1:
        # use default name for the parameter file
        print(sys.argv)
        pars = parser.parse_args([default_parameter_file])
    else:
        pars = parser.parse_args()

    # label_dict = {'Nucleus': 1, 'Her2': 2, 'CEP17': 3}
    # category_index = {1: {'id': 1, 'name': 'Nucleus'}, 2: {'id': 2, 'name': 'Her2'}, 3: {'id': 3, 'name': 'CEP17'}}

    label_id_offset = 1
    num_classes = len(pars.label_dict)
    category_index = {}
    for key, value in pars.label_dict.items():
        category_index[value] = {'id': value, 'name': key}

    # model_config = \
    #     '/home/idekany/Research/codes/models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'

    # initial_checkpoint = \
    #     '/home/idekany/Research/codes/models/research/object_detection/checkpoints/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0'

    # load_checkpoint = "early_stopping_checkpoints/early_stopping_ckpt-1"

    # train_checkpoint_dir = "checkpoints"

    # early_stopping_checkpoint_dir = "early_stopping_checkpoints"

    # WIRED-IN PARAMETERS:

    train_shuffle_buffer_size = 18
    val_shuffle_buffer_size = 4
    max_total_detections = 100
    max_detections_per_class = 100
    iou_threshold = 0.5                 # IOU threshold in the computation of the mAP

    if pars.optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=pars.learning_rate)
    elif pars.optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=pars.learning_rate, momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=pars.learning_rate, momentum=0.9)

    # ----------------------------------------------------------------------------------------------------------------------
    # Check the number of GPUs and set identical memory growth:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # check if there are any GPUs
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("Number of GPUs:", len(gpus), "physical,", len(logical_gpus), "logical")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

        n_gpus = len(gpus)
    else:
        n_gpus = 0

    # ----------------------------------------------------------------------------------------------------------------------
    """
    Instantiate the model to be fine-tuned.
    """

    tf.keras.backend.clear_session()

    print('Building model and restoring weights for fine-tuning...', flush=True)

    retinanet_model = RetinaNetModel(pars.model_config,
                                     optimizer,
                                     checkpoint_dir=pars.train_checkpoint_dir,
                                     load_checkpoint=pars.load_checkpoint,
                                     num_classes=num_classes,
                                     initialize=pars.initialize_training,
                                     init_classification_head=pars.init_classification_head,
                                     init_regression_head=pars.init_regression_head,
                                     init_checkpoint=pars.initial_checkpoint,
                                     iou_threshold=iou_threshold,
                                     min_fpn_level=pars.min_fpn_level,
                                     max_fpn_level=pars.max_fpn_level,
                                     anchor_scale=pars.anchor_scale,
                                     scales_per_octave=pars.scales_per_octave,
                                     max_total_detections=max_total_detections,
                                     max_detections_per_class=max_detections_per_class)

    # ----------------------------------------------------------------------------------------------------------------------
    """
    Load the lists of image paths, ground truth boxes, and labels from json files:
    """

    train_image_list, train_gt_annot, train_gt_labels = \
        load_json_images_annotations_from_list(
            pars.train_image_list_file,
            pars.label_dict,
            subdir=pars.train_image_dir,
            annot_suffix=pars.annotation_file_suffix)

    val_image_list, val_gt_annot, val_gt_labels = \
        load_json_images_annotations_from_list(
            pars.val_image_list_file,
            pars.label_dict,
            subdir=pars.val_image_dir,
            annot_suffix=pars.annotation_file_suffix)

    n_train_images = len(train_image_list)
    n_val_images = len(val_image_list)

    # ----------------------------------------------------------------------------------------------------------------------
    """
    Prepare datasets for training and validation.
    """

    # Define functions for data augmentation:

    random_crop_image = ia.get_crop_func(
        target_height=640, target_width=640)

    random_flip = ia.get_flip_func(
        prob_horizontal=0.5, prob_vertical=0.5)

    random_change_image = ia.get_change_image_func(
        max_brightness_delta=0.2, lower_contrast_factor=0.8, upper_contrast_factor=1.0,
        lower_saturation_factor=0.75, upper_saturation_factor=1.25, max_hue_delta=0.05)

    random_rotate = ia.get_rotate_func(
        prob_rot=1, max_rot_angle=tf.constant(np.pi / 15.), crop=True)

    preprocess = ia.get_preprocess_func(retinanet_model.detection_model)


    # Create training dataset:

    train_datagen = get_datagen(train_image_list, train_gt_annot, train_gt_labels, num_classes, label_id_offset)

    train_dataset = tf.data.Dataset.from_generator(
        train_datagen,
        output_signature=(
            tf.RaggedTensorSpec(shape=(None, None, 3), ragged_rank=1, dtype=tf.float32, row_splits_dtype=tf.int32),
            tf.TensorSpec(shape=(3,), dtype=tf.int32),
            tf.RaggedTensorSpec(shape=(None, 5), dtype=tf.float32, row_splits_dtype=tf.int32),
            tf.RaggedTensorSpec(shape=(None, None), dtype=tf.float32, row_splits_dtype=tf.int32))
    )

    train_dataset = train_dataset \
        .cache() \
        .map(random_rotate, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(random_crop_image, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(random_flip, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(random_change_image, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(buffer_size=train_shuffle_buffer_size) \
        .map(preprocess, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(ia.ell2box, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(pars.batch_size)

    # Check data sanity by plotting a training batch:

    # data_batch = train_dataset.batch(batch_size).take(1)
    #
    # plot_image_batch_with_ellipses(data_batch, category_index, ell2box=True,
    #                                figname='image_batch_with_ellipses_crop', figformat='pdf')

    plot_image_batch_with_boxes(train_dataset.take(1), category_index, label_id_offset,
                                figname='image_tr_batch_with_boxes', figformat='pdf')


    # Create validation dataset:

    val_datagen = get_datagen(val_image_list, val_gt_annot, val_gt_labels, num_classes, label_id_offset)

    val_dataset = tf.data.Dataset.from_generator(
        val_datagen,
        output_signature=(
            tf.RaggedTensorSpec(shape=(None, None, 3), ragged_rank=1, dtype=tf.float32, row_splits_dtype=tf.int32),
            tf.TensorSpec(shape=(3,), dtype=tf.int32),
            tf.RaggedTensorSpec(shape=(None, 5), dtype=tf.float32, row_splits_dtype=tf.int32),
            tf.RaggedTensorSpec(shape=(None, None), dtype=tf.float32, row_splits_dtype=tf.int32))
    )

    val_dataset = val_dataset \
        .cache() \
        .map(random_rotate, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(random_crop_image, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(random_flip, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(random_change_image, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(buffer_size=val_shuffle_buffer_size) \
        .map(preprocess, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(ia.ell2box, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(pars.val_batch_size)

    # Check data sanity by plotting a validation batch:

    plot_image_batch_with_boxes(train_dataset.take(1), category_index, label_id_offset,
                                figname='image_val_batch_with_boxes', figformat='pdf')

    # ----------------------------------------------------------------------------------------------------------------------
    """
    Eager mode training loop.
    """

    print('Fine-tuning the model...', flush=True)

    # Set up progress bar for epochs:

    epochs = trange(pars.num_epochs, desc="Epoch", unit="epoch", position=0, leave=True,
                    postfix="mean_loss = {mean_loss:.4f}, "
                            "mean_map = {mean_map:.4f}, "
                            "mean_iou = {mean_iou:.4f}, "
                            "val_mean_loss = {val_mean_loss:.4f}, "
                            "val_mean_map = {val_mean_map:.4f}, "
                            "val_mean_iou = {val_mean_iou:.4f}"
                            "val_mean_scores_maxf1 = {val_mean_scores_maxf1}")
    epochs.set_postfix(mean_loss=None, mean_map=None, mean_iou=None,
                       val_mean_loss=None, val_mean_map=None, val_mean_iou=None,
                       val_mean_scores_maxf1=None)

    # Set up callback list:

    _callback_list = [
        PlotLearning(eval_metrics=['iou', 'map'], n_zoom=20, n_update=1, figname="liveplot"),
        EarlyStopping(monitor='val_loss', min_delta=10e-5, patience=pars.early_stopping_patience, verbose=1,
                      mode='min', baseline=None, restore_best_weights=True, checkpoint_best_weights=True,
                      checkpoint_dir=pars.early_stopping_checkpoint_dir, optimizer=optimizer)
    ]
    callbacks = tf.keras.callbacks.CallbackList(_callback_list, add_history=True, model=retinanet_model.detection_model)


    # START TRAINING LOOP:

    logs = {}
    callbacks.on_train_begin(logs=logs)

    for i_epoch in epochs:

        # Set up progress bar for batches:
        batches = tqdm(enumerate(train_dataset), desc="Batch", unit="batch",
                       postfix="mean_batch_loss = {mean_batch_loss:.4f}, mean_loss = {mean_loss:.4f}",
                       position=1, leave=False)
        batches.set_postfix(mean_batch_loss=None, mean_loss=None)

        tf.keras.backend.set_learning_phase(1)

        sum_of_batch_losses = 0
        mean_loss = 0
        iou_list = []
        map_list = []

        for i_batch, train_batch in batches:

            # Get lists of tensors from current dataset batch:
            image_tensor_batch, gt_boxes_rtensor_batch, gt_classes_rtensor_batch = train_batch
            gt_boxes_list = get_list_from_ragged_batch(gt_boxes_rtensor_batch)
            gt_classes_list = get_list_from_ragged_batch(gt_classes_rtensor_batch)

            # Training step:
            train_detections, total_batch_loss, map_batch, mean_iou_batch, scores_maxf1 = \
                retinanet_model.train_step(image_tensor_batch, gt_boxes_list, gt_classes_list)

            plot_images_with_boxes(image_tensor_batch, gt_boxes_list, gt_classes_list,
                                   train_detections['detection_boxes'],
                                   train_detections['detection_classes'],
                                   train_detections['detection_scores'],
                                   category_index, label_id_offset,
                                   figname='training_batch_with_boxes', figformat='jpg',
                                   min_score_thresh=pars.min_score_thresh,
                                   max_boxes_to_draw=20,
                                   skip_scores=False,
                                   skip_labels=False)

            total_batch_loss = total_batch_loss.numpy()

            # Update mean batch loss of this epoch with the loss from the current batch:
            sum_of_batch_losses += total_batch_loss
            # mean loss per image computed from all batches processed so far:
            mean_loss = sum_of_batch_losses / ((i_batch + 1) * pars.batch_size)
            # mean loss per image computed from this batch only:
            mean_batch_loss = total_batch_loss / pars.batch_size
            # print(i_batch, total_batch_loss, sum_of_batch_losses, mean_batch_loss)
            iou_list.append(mean_iou_batch)
            map_list.append(map_batch)
            batches.set_postfix(mean_batch_loss=mean_batch_loss,
                                mean_loss=mean_loss)

        mean_iou = tf.reduce_mean(iou_list).numpy()
        mean_map = tf.reduce_mean(map_list).numpy()
        logs['map'] = mean_map
        logs['iou'] = mean_iou
        logs['loss'] = mean_loss

        # End of epoch:
        tf.keras.backend.set_learning_phase(0)

        val_batches = tqdm(enumerate(val_dataset), desc="Val. batch", unit="batch",
                           postfix="mean_val_batch_loss = {mean_val_batch_loss:.4f}, "
                                   "val_mean_loss = {val_mean_loss:.4f}",
                           position=1, leave=False)
        val_batches.set_postfix(loss=None, mean_loss=None)
        val_iou_list = []
        val_map_list = []
        scores_maxf1_list = []
        sum_of_val_batch_losses = 0
        val_mean_loss = 0
        for i_batch, val_batch in val_batches:
            image_tensor_batch, gt_boxes_rtensor_batch, gt_classes_rtensor_batch = val_batch
            gt_boxes_list = get_list_from_ragged_batch(gt_boxes_rtensor_batch)
            gt_classes_list = get_list_from_ragged_batch(gt_classes_rtensor_batch)

            val_detections, total_val_batch_loss, val_map_batch, val_mean_iou_batch, scores_maxf1 = \
                retinanet_model.val_step(image_tensor_batch, gt_boxes_list, gt_classes_list)

            plot_images_with_boxes(image_tensor_batch, gt_boxes_list, gt_classes_list,
                                   val_detections['detection_boxes'],
                                   val_detections['detection_classes'],
                                   val_detections['detection_scores'],
                                   category_index, label_id_offset, min_score_thresh=pars.min_score_thresh,
                                   figname='validation_batch_with_boxes', figformat='jpg',
                                   max_boxes_to_draw=20, skip_scores=False, skip_labels=False)

            total_val_batch_loss = total_val_batch_loss.numpy()
            sum_of_val_batch_losses += total_val_batch_loss
            # mean loss per image computed from all batches processed so far:
            val_mean_loss = sum_of_val_batch_losses / ((i_batch + 1) * pars.val_batch_size)
            # mean loss per image computed from this batch only:
            mean_val_batch_loss = total_val_batch_loss / pars.val_batch_size

            val_iou_list.append(val_mean_iou_batch)
            val_map_list.append(val_map_batch)
            scores_maxf1_list.append(scores_maxf1)
            # print(scores_maxf1)
            val_batches.set_postfix(mean_val_batch_loss=mean_val_batch_loss,
                                    val_mean_loss=val_mean_loss)

        val_mean_iou = tf.reduce_mean(val_iou_list).numpy()
        val_mean_map = tf.reduce_mean(val_map_list).numpy()
        val_mean_scores_maxf1 = tf.reduce_mean(tf.stack(scores_maxf1_list), axis=0).numpy()
        # print(mean_scores_maxf1)
        logs['val_iou'] = val_mean_iou
        logs['val_loss'] = val_mean_loss
        logs['val_map'] = val_mean_map

        epochs.set_postfix(mean_loss=mean_loss, mean_map=mean_map, mean_iou=mean_iou,
                           val_mean_loss=val_mean_loss, val_mean_map=val_mean_map, val_mean_iou=val_mean_iou,
                           val_mean_scores_maxf1=str(val_mean_scores_maxf1))

        callbacks.on_epoch_end(i_epoch, logs=logs)
        if retinanet_model.detection_model.stop_training:
            break

        if pars.save_checkpoint and (i_epoch + 1) % pars.checkpoint_freq == 0:
            retinanet_model.save_checkpoint()

    callbacks.on_train_end(logs=logs)

    print('Done fine-tuning!')
    # ----------------------------------------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------------------------------------
    """
    Do inference using the (fine-tuned) model on a set of test images.
    """

    if pars.run_test:

        tf.keras.backend.set_learning_phase(0)

        with open(pars.test_image_list_file) as f:
            test_image_list = np.loadtxt(f, dtype='str')

        for test_image in test_image_list:

            image_path = os.path.join(pars.test_image_dir, test_image)

            test_image_np = np.expand_dims(load_image_into_numpy_array(image_path), axis=0)
            input_tensor = tf.convert_to_tensor(test_image_np, dtype=tf.float32)

            detections = retinanet_model.detect(input_tensor)

            plot_detections(
                test_image_np[0],
                detections['detection_boxes'][0].numpy(),
                detections['detection_classes'][0].numpy().astype(np.uint32) + label_id_offset,
                detections['detection_scores'][0].numpy(),
                category_index,
                min_score_thresh=pars.min_score_thresh,
                image_name=os.path.join(pars.test_image_dir, 'detections_' + test_image))


if __name__ == '__main__':

    main()
