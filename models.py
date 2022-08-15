import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from utils import *
import sys

class RetinaNetModel:
    """
    Encompasses a pre-trained RetinaNet model and the methods to fine-tune it.
    """
    def __init__(self,
                 config_handle,
                 optimizer,
                 input_image_size=640,
                 num_classes=1,
                 initialize=True,
                 init_checkpoint=None,
                 init_classification_head=True,
                 init_regression_head=False,
                 checkpoint_dir='.',
                 max_checkpoints_to_keep=3,
                 load_checkpoint='ckpt-0',
                 iou_threshold=0.5,
                 min_fpn_level=3,
                 max_fpn_level=7,
                 anchor_scale=4.0,
                 scales_per_octave=2,
                 max_total_detections=100,
                 max_detections_per_class=100,
                 n_scores=100):
        """
        Constructor of a custom RetinaNet model.

        :param config_handle: str
            Path to the detection model's configuration file.
        :param optimizer: object
            A tensorflow.keras optimizer model instance. We include this in the class in order to
            be able to save it in the model checkpoints.
        :param input_image_size: int
            The fixed square image size at the model's input. Must be consistent with the configuration file,
            and be either 640 or 1024.
        :param num_classes: int
            The number of class labels to expect in the training set.
        :param initialize: bool
            Whether to initialize the model with the published pre-trained weights.
        :param init_checkpoint: str
            Path to the checkpoint for weight initialization. If None, it has the same effect as initialize=False.
        :param init_classification_head: bool
            Whether to initialize the model's classification head. If False, its pre-trained weights will be restored.
        :param init_regression_head:
            Whether to initialize the model's box prediction head. If False, its pre-trained weights will be restored.
        :param checkpoint_dir: str
            Directory path for storing saved model checkpoints.
        :param max_checkpoints_to_keep: int
            The maximum number of model checkpoints to keep. When this number is reached, the oldest checkpoint
            will be overwritten with the latest one.
        :param load_checkpoint: str
            The path to an earlier checkpoint to be restored. If `init_checkpoint` is False, a checkpoint saved
            from an earlier training round is expected.
        :param iou_threshold: float
            The Intersection Over Union threshold to be used in the computation of the mean Average Precision score.
            Must be in the (0,1) interval.
        :param min_fpn_level: int
            The minimum level of the backbone netork to be used in the Feature Pyramid Network.
        :param max_fpn_level: int
            The maximum level of the backbone netork to be used in the Feature Pyramid Network.
        :param anchor_scale: float
            Base anchor scale at a single level.
        :param scales_per_octave: int
            The number of anchor scales at each level.
            The anchor scales for anchors in a single level are computed as follows:
            anchor_scales = 2**(i/scales_per_octave) * anchor_scale, for i = 1,...,scales_per_octave
        :param max_total_detections: int
            The maximum number of detections for all classes combined.
        :param max_detections_per_class: int
            The maximum number of detections per class.
        :param n_scores: int
            The number of incremental probability scores to be used in the computation of the mean Average Precision.
        """

        configs = config_util.get_configs_from_pipeline_file(config_handle)
        self.model_config = configs['model']
        self.model_config.ssd.num_classes = num_classes
        self.model_config.ssd.freeze_batchnorm = True
        self.model_config.ssd.anchor_generator.multiscale_anchor_generator.min_level = \
            min_fpn_level
        self.model_config.ssd.anchor_generator.multiscale_anchor_generator.max_level = \
            max_fpn_level
        self.model_config.ssd.anchor_generator.multiscale_anchor_generator.anchor_scale = \
            anchor_scale
        self.model_config.ssd.anchor_generator.multiscale_anchor_generator.scales_per_octave = \
            scales_per_octave
        self.model_config.ssd.post_processing.batch_non_max_suppression.max_total_detections = \
            max_total_detections
        self.model_config.ssd.post_processing.batch_non_max_suppression.max_detections_per_class = \
            max_detections_per_class

        self.detection_model = model_builder.build(model_config=self.model_config, is_training=True)
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.load_checkpoint = load_checkpoint

        assert 0 < iou_threshold < 1, \
            "Invalid parameter: iou_threshold={}. Value must be in the (0,1) interval.".format(iou_threshold)
        self.iou_threshold = iou_threshold

        assert n_scores >= 10 and type(n_scores) == int, \
            "Invalid parameter: n_scores={}. Value must be an integer >= 10.".format(n_scores)
        self.n_scores = n_scores

        assert num_classes > 0 and type(num_classes) == int, \
            "Invalid parameter: num_classes={}. Value must be a non-negative integer.".format(num_classes)
        self.num_classes = num_classes

        assert input_image_size in (640, 1024), \
            "Invalid parameter: input_image_size={}. Value must be 640 or 1024.".format(input_image_size)
        if str(input_image_size) not in config_handle:
            print("Warning: parameter `input_image_size` seems to be inconsistent with `config_handle`. "
                  "Check your model parameter settings.", file=sys.stderr)
        self.input_image_size = input_image_size

        self.prefixes_to_train = \
            ['WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
             'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']

        if initialize and init_checkpoint is not None:

            # The model has two prediction heads (classification and bounding box regression).
            # Here, we restore the weight of the backbone resnet and the base tower layers for the prediction heads.
            # Either prediction heads can be restred or initialized, depending on the transfer learning task.

            if init_classification_head and init_regression_head:
                # Initialize both regression and classification heads, restore the base tower layers only.
                prediction_heads = tf.compat.v2.train.Checkpoint(
                    _base_tower_layers_for_heads=self.detection_model._box_predictor._base_tower_layers_for_heads,
                )
            elif init_classification_head:
                # Restore regression head only:
                prediction_heads = tf.compat.v2.train.Checkpoint(
                    _base_tower_layers_for_heads=self.detection_model._box_predictor._base_tower_layers_for_heads,
                    _box_prediction_head=self.detection_model._box_predictor._box_prediction_head,
                )

            elif init_regression_head:
                # Restore classification head only:
                prediction_heads = tf.compat.v2.train.Checkpoint(
                    _base_tower_layers_for_heads=self.detection_model._box_predictor._base_tower_layers_for_heads,
                    _prediction_heads=self.detection_model._box_predictor._prediction_heads,
                )

            else:
                # Restore both classification and regression heads:
                prediction_heads = tf.compat.v2.train.Checkpoint(
                    _base_tower_layers_for_heads=self.detection_model._box_predictor._base_tower_layers_for_heads,
                    _prediction_heads=self.detection_model._box_predictor._prediction_heads,
                    _box_prediction_head=self.detection_model._box_predictor._box_prediction_head,
                )

            temp_model = tf.compat.v2.train.Checkpoint(
                _feature_extractor=self.detection_model._feature_extractor,
                _box_predictor=prediction_heads)
            ckpt = tf.compat.v2.train.Checkpoint(model=temp_model)
            ckpt.restore(init_checkpoint).expect_partial()

            # Run an empty image through the model, so that variables are created:
            image, shapes = \
                self.detection_model.preprocess(tf.zeros([1, self.input_image_size, self.input_image_size, 3]))
            prediction_dict = self.detection_model.predict(image, shapes)
            _ = self.detection_model.postprocess(prediction_dict, shapes)

            print("Weights initialized from: \n\t{}".format(init_checkpoint))

        else:

            assert self.load_checkpoint, "Parameter `load_checkpoint` must be specified or initialize must be True."

            ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model, optimizer=self.optimizer)
            # Load latest checkpoint with specified name `self.load_checkpoint`.
            print("Restoring weights from {}".format(load_checkpoint))
            ckpt.restore(load_checkpoint)
            # Run a dummy image through the model, so that variables are created
            image, shapes = \
                self.detection_model.preprocess(tf.zeros([1, self.input_image_size, self.input_image_size, 3]))
            prediction_dict = self.detection_model.predict(image, shapes)
            _ = self.detection_model.postprocess(prediction_dict, shapes)

        # Instantiate checkpoint manager to save regular checkpoints with default name:
        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir,
                                                  checkpoint_name='ckpt', max_to_keep=max_checkpoints_to_keep)

        # Select variables in top layers to fine-tune.
        trainable_variables = self.detection_model.trainable_variables
        self.tuned_variables = []

        for var in trainable_variables:
            if any([var.name.startswith(prefix) for prefix in self.prefixes_to_train]):
                self.tuned_variables.append(var)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, image_tensor_batch, gt_boxes_list, gt_classes_list):
        """A single training iteration.

        Args:
        :param image_tensor_batch: tf.float32
            A tf.Tensor of shape (batch_size, height, width, 3), representing a batch of training images.
            The height and width must not vary across the images, i.e. a ragged tensor is not accepted.
            The images that form this tensor are supposed to be preprocessed.
        :param gt_boxes_list: list of tf.float32 tensors
            Each element of the list is of shape (N_objects, 4), representing the groundtruth boxes
            for each image in `image_tensor_batch`.
        :param gt_classes_list: list of tf.float32 tensors
            Each element of the list is of shape (N_objects, N_classes), representing the one-hot encoded
            groundtruth classes of the object boxes in each image in `image_tensor_batch`.
        :return: total_loss, mean_iou_batch
            total_loss: A scalar tf.Tensor representing the total loss for the input mini-batch.
            mean_iou_batch: IOU metric averaged over each image, class, and box in the input mini-batch.

        Returns:
          A scalar tensor representing the total loss for the input batch.
        """

        batch_size = image_tensor_batch.shape[0]

        shapes = tf.constant(batch_size * [[self.input_image_size, self.input_image_size, 3]], dtype=tf.int32)
        self.detection_model.provide_groundtruth(
            groundtruth_boxes_list=gt_boxes_list,
            groundtruth_classes_list=gt_classes_list)
        with tf.GradientTape() as tape:
            prediction_dict = self.detection_model.predict(image_tensor_batch, shapes)
            detections = self.detection_model.postprocess(prediction_dict, shapes)
            losses_dict = self.detection_model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, self.tuned_variables)
            self.optimizer.apply_gradients(zip(gradients, self.tuned_variables))

        map_batch, mean_iou_batch, scores_maxf1 = \
            compute_map_iou_per_batch(gt_boxes_list, gt_classes_list, detections, batch_size,
                                      iou_threshold=self.iou_threshold, n_scores=self.n_scores,
                                      num_classes=self.num_classes)

        return detections, total_loss, map_batch, mean_iou_batch, scores_maxf1

    @tf.function(experimental_relax_shapes=True)
    def val_step(self, image_tensors, gt_boxes_list, gt_classes_list):
        """
        A single validation iteration on a mini-batch of data.

        :param image_tensors: tf.float32
            A tf.Tensor of shape (batch_size, height, width, 3), representing a batch of training images.
            The height and width must not vary across the images, i.e. a ragged tensor is not accepted.
            The images that form this tensor are supposed to be preprocessed.
        :param gt_boxes_list: list of tf.float32 tensors
            Each element of the list is of shape (N_objects, 4), representing the groundtruth boxes
            for each image in `image_tensor_batch`.
        :param gt_classes_list: list of tf.float32 tensors
            Each element of the list is of shape (N_objects, N_classes), representing the one-hot encoded
            groundtruth classes of the object boxes in each image in `image_tensor_batch`.
        :return: total_loss, mean_iou_batch
            total_loss: A scalar tf.Tensor representing the total loss for the input mini-batch.
            mean_iou_batch: IOU metric averaged over each image, class, and box in the input mini-batch.
        """
        batch_size = len(image_tensors)

        shapes = tf.constant(batch_size * [[self.input_image_size, self.input_image_size, 3]], dtype=tf.int32)
        self.detection_model.provide_groundtruth(
            groundtruth_boxes_list=gt_boxes_list,
            groundtruth_classes_list=gt_classes_list)
        prediction_dict = self.detection_model.predict(image_tensors, shapes)
        val_detections = self.detection_model.postprocess(prediction_dict, shapes)
        losses_dict = self.detection_model.loss(prediction_dict, shapes)

        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

        map_batch, mean_iou_batch, scores_maxf1 = \
            compute_map_iou_per_batch(gt_boxes_list, gt_classes_list, val_detections, batch_size,
                                      iou_threshold=self.iou_threshold, n_scores=self.n_scores,
                                      num_classes=self.num_classes)

        return val_detections, total_loss, map_batch, mean_iou_batch, scores_maxf1

    @tf.function
    def detect(self, input_tensor):
        preprocessed_image, shapes = self.detection_model.preprocess(input_tensor)
        prediction_dict = self.detection_model.predict(preprocessed_image, shapes)
        return self.detection_model.postprocess(prediction_dict, shapes)

    def save_checkpoint(self):
        self.manager.save()
