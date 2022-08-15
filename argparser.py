import argparse


default_parameter_file = '@settings.par'


def argparser():
    """
    Creates an argparse.ArgumentParser object for reading in parameters from a file.
    :return: object
        An argparse.ArgumentParser object.
    """
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                 description='Fine-tune a pre-trained RetinaNet object detection model '
                                             'on a custom data set by transfer learning.',
                                 epilog="")

    # use a custom line parser function for the parameter file:
    ap.convert_arg_line_to_args = convert_arg_line_to_args

    ap.add_argument("--label_dict",
                    dest="label_dict",
                    action=StoreDictKeyValuePair,
                    required=True,
                    nargs="+",
                    metavar="KEY1=VAL1 KEY2=VAL2",
                    help='Key-value pairs of the classes, where the keys are class names and the values are 1-based '
                         'integer labels.')

    ap.add_argument('--num_epochs',
                    action='store',
                    type=int,
                    default=1,
                    help='The number of training epochs to run.')

    ap.add_argument('--batch_size',
                    action='store',
                    type=int,
                    default=2,
                    help='The training batch size.')

    ap.add_argument('--learning_rate',
                    action='store',
                    type=float,
                    default=0.01,
                    help='Learning rate.')

    ap.add_argument('--val_batch_size',
                    action='store',
                    type=int,
                    default=4,
                    help='The validation batch size.')

    ap.add_argument('--optimizer',
                    action='store',
                    type=str,
                    default='SGD',
                    help='The optimization algorithm to be used for training. '
                         'Current options are: SGD, Adam.')

    ap.add_argument('--model_config',
                    action='store',
                    type=str,
                    required=True,
                    help='Absolute path to the model configuration file.')

    ap.add_argument('--initialize_training',
                    action='store_true',
                    help="Whether to initialize the model with the published pre-trained weights before training.")

    ap.add_argument('--initial_checkpoint',
                    action='store',
                    type=str,
                    default=None,
                    help="Absolute path to the model's initialization checkpoint.")

    ap.add_argument('--init_classification_head',
                    action='store_true',
                    help="Whether to initialize the model's classification head if initialize_training is set. "
                         "If not set, its parameters will be restored instead.")

    ap.add_argument('--init_regression_head',
                    action='store_true',
                    help="Whether to initialize the model's box regression head if initialize_training is set. "
                         "If not set, its parameters will be restored instead.")

    ap.add_argument('--load_checkpoint',
                    action='store',
                    type=str,
                    default=None,
                    help="Path to the model's last checkpoint in case of retraining (if initialize_training=False).")

    ap.add_argument('--save_checkpoint',
                    action='store_true',
                    help="Whether to save checkpoint periodically.")

    ap.add_argument('--checkpoint_freq',
                    action='store',
                    type=int,
                    default=10,
                    help="Frequency (in number of epochs) for saving checkpoints periodically.")

    ap.add_argument('--train_checkpoint_dir',
                    action='store',
                    type=str,
                    default='.',
                    help="Subdirectory for periodically saved checkpoints.")

    ap.add_argument('--early_stopping_patience',
                    action='store',
                    type=int,
                    default=10,
                    help="Number of training epochs to wait for the validation performance to improve "
                         "before early stopping is performed.")

    ap.add_argument('--early_stopping_checkpoint_dir',
                    action='store',
                    type=str,
                    default='.',
                    help="Subdirectory for early stopping checkpoints.")

    ap.add_argument('--train_image_dir',
                    action='store',
                    type=str,
                    default='data',
                    help='Relative path of the subdirectory of training images.')

    ap.add_argument('--train_image_list_file',
                    action='store',
                    type=str,
                    default='training_images.lst',
                    help='File with the list of training images.')

    ap.add_argument('--val_image_dir',
                    action='store',
                    type=str,
                    default='data',
                    help='Relative path of the subdirectory of validation images.')

    ap.add_argument('--val_image_list_file',
                    action='store',
                    type=str,
                    default='validation_images.lst',
                    help='File with the list of training images.')

    ap.add_argument('--test_image_dir',
                    action='store',
                    type=str,
                    default='data',
                    help='Relative path of the subdirectory of test images.')

    ap.add_argument('--run_test',
                    action='store_true',
                    help='Whether to run inference on a list of test images.')

    ap.add_argument('--test_image_list_file',
                    action='store',
                    type=str,
                    default='test_images.lst',
                    help='File with the list of test images.')

    ap.add_argument('--annotation_file_suffix',
                    action='store',
                    type=str,
                    default='_annot.json',
                    help='Suffix for the annotation files in the scheme of: myimage.png -> myimage_annot.json')

    ap.add_argument('--min_score_thresh',
                    action='store',
                    type=float,
                    default=0.5,
                    help='Minimum detection score threshold for plotting a detection box.')

    ap.add_argument('--min_fpn_level',
                    action='store',
                    type=int,
                    default=3,
                    help='The minimum level of the backbone netork to be used in the Feature Pyramid Network.')

    ap.add_argument('--max_fpn_level',
                    action='store',
                    type=int,
                    default=7,
                    help='The maximum level of the backbone netork to be used in the Feature Pyramid Network.')

    ap.add_argument('--anchor_scale',
                    action='store',
                    type=float,
                    default=4.0,
                    help='Base anchor scale at a single level.')

    ap.add_argument('--scales_per_octave',
                    action='store',
                    type=int,
                    default=2,
                    help='The number of anchor scales at each level. '
                    'The anchor scales for anchors in a single level are computed as follows: '
                    'anchor_scales = 2**(i/scales_per_octave) * anchor_scale, for i = 1,...,scales_per_octave')

    return ap


def convert_arg_line_to_args(arg_line):
    """
    Custom line parser for argparse.
    :param arg_line: str
    One line of the input parameter file.
    :return: None
    """
    if arg_line:
        if arg_line[0] == '#':
            return
        for arg in arg_line.split():
            if not arg.strip():
                continue
            if '#' in arg:
                break
            yield arg


class StoreDictKeyValuePair(argparse.Action):
    """
    Custom argparse class for storing dictionary key-value pairs.
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyValuePair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = {}
        for kv in values:
            k, v = kv.split("=")
            input_dict[k] = int(v)
        setattr(namespace, self.dest, input_dict)
