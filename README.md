# TuneRetinaNet

This is a custom python / TensorFlow library for applying transfer learning on a RetinaNet
object detection model with image augmentation.


## Installation

Simply use `git clone <URL>` to clone this library into a local directory. 
Subsequently, it can be added to your `PYTHONPATH` either temporarily by 
issuing the `sys.path.append("/your/local/directory")` command in python,
or permanantly by exporting you directory into the `PYTHONPATH` system variable.
For example, if using the bash shell, add the 
`export PYTHONPATH="${PYTHONPATH}:/your/local/directory"` in the ~/.bashrc
file.

### Dependencies

Lcfit requires python 3.8 and the 
[TensorFlow](https://tensorflow.org),
[TensorFlow Addons](https://www.tensorflow.org/addons),
[numpy](https://numpy.org/), 
[scikit-learn](https://scikit-learn.org/stable/), and
[matplotlib](https://matplotlib.org/)
python libraries for correct functionality.

## Usage
To fine-tune a RetinaNet model on a custom data set, run the `train_detector.py` 
module using command-line arguments:
`python train_detector.py [OPTION]`,
or by supplying a parameter file that includes the command-line arguments:
`python train_detector.py @<parameter_file>`.
The full list of command-line options can be printed on the STDOUT by:
`python train_detector.py --help`.

The configuration file and initial checkpoint files are not included in this repository,
and can be downloaded from, e.g. the 
[TensorFlow Model Garden](https://github.com/tensorflow/models)

The program expects JSON annotation files in the output format of the
[VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/).
Furthermore, the `train_detector.py` module requires ellipse annotations.

## License

[MIT](https://choosealicense.com/licenses/mit/), see `LICENSE`.
