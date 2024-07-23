# MaLDyDVS: Machine Learning for Dynamic Vision Sensors

MaLDyVS is a Machine Learning package for Dynamic Vision Sensors using PyTorch. It proposes some tools for event-based data: event visualisation, and event representations to use standard  machine and deep learning methods.

MaLDyVS is supposed to be an all set-in package, where most of the (annoying) things have been
already developed in order to accelerate development for event-based research. Still, it is
designed to be fully adjustable for your own personal requirements, whatever you're working on.

MaLDyVS proposes built-in functions and classes to ease development, training, and evaluation
of event-based approaches in pure Python, while exploiting the excellent performances of GPUs,
thanks to [PyTorch](https://pytorch.org/) and [Torchvision](https://pytorch.org/vision/stable/index.html).

News: the code will be available soon.

## Why MLDVS ?
1. Fully **open-source** project.
2. Entirely **built around [PyTorch]()**, and built as an extension of [Torchvision]() for event-based approaches.
3. **Flexibility**: With the same package, different script setups or installation are easy to handle thanks to configuration files (see for instance [config.ini](config.ini) to set up the working environment, or [json files for model parameters](model_conf/event_baseline_config.json))
4. **Easy to use, and to extend.**


## Summary
- [Installation](https://github.com/LaureAcin/MaLDyVS#installation)
- [Documentation Generation](https://github.com/LaureAcin/MaLDyVS#documentation-generation)
- [Few examples](https://github.com/LaureAcin/MaLDyVS#few-examples)
  - [Event visualisation on DVS128 Gesture dataset](https://github.com/LaureAcin/MaLDyVS#event-visualisation-on-DVS128-Gesture-dataset)
  - [Event-based recognition using using DVS128-Gesture dataset and VK-SITS event representation](https://github.com/LaureAcin/MaLDyVS#Event-based-recognition-using-DVS128-Gesture-dataset-and-VK-SITS-event-representation)
- [Evaluation](https://github.com/LaureAcin/MaLDyVS#evaluation)
- [Documentation](https://github.com/LaureAcin/MaLDyVS#Documentation)
    - [Available datasets](https://github.com/LaureAcin/MaLDyVS#available-datasets)
    - [Available event-visualisation methods](https://github.com/LaureAcin/MaLDyVS#Available-event-visualisation-methods)
      - [disp_video](https://github.com/LaureAcin/MaLDyVS#disp_video)
      - [disp_decay_video](https://github.com/LaureAcin/MaLDyVS#disp_decay_video)
      - [disp3D](https://github.com/LaureAcin/MaLDyVS#disp3D)
      - [Event representations](https://github.com/LaureAcin/MaLDyVS#Event-representations)
      - [EST tensor](https://github.com/LaureAcin/MaLDyVS#EST-tensor)
        - [EventCounting](https://github.com/LaureAcin/MaLDyVS#EventCounting)
      - [HOTS](https://github.com/LaureAcin/MaLDyVS#HOTS)
        - [AsyncTimeSurface](https://github.com/LaureAcin/MaLDyVS#HOTS)
        - [TimeSurface](https://github.com/LaureAcin/MaLDyVS#HOTS)
      - [HATS](https://github.com/LaureAcin/MaLDyVS#HATS)
        - [AsyncLocalMemoryTimeSurface](https://github.com/LaureAcin/MaLDyVS#AsyncLocalMemoryTimeSurface)
        - [LocalMemoryTimeSurface](https://github.com/LaureAcin/MaLDyVS#LocalMemoryTimeSurface)
      - [HATS using FIFO implementation](https://github.com/LaureAcin/MaLDyVS#HATS-using-FIFO-implementation)
        - [AsyncFifoHATS](https://github.com/LaureAcin/MaLDyVS#HATS-using-FIFO-implementation)
        - [FifoHATS](https://github.com/LaureAcin/MaLDyVS#HATS-using-FIFO-implementation)
      - [TORE](https://github.com/LaureAcin/MaLDyVS#TORE)
        - [AsyncTORE](https://github.com/LaureAcin/MaLDyVS#TORE)
        - [TORE](https://github.com/LaureAcin/MaLDyVS#TORE)
      - [SITS](https://github.com/LaureAcin/MaLDyVS#SITS)
        - [AsyncSITS](https://github.com/LaureAcin/MaLDyVS#AsyncSITS)
        - [SITS](https://github.com/LaureAcin/MaLDyVS#SITS)
      - [VoxelGrid](https://github.com/LaureAcin/MaLDyVS#VoxelGrid)
        - [VoxelGrid](https://github.com/LaureAcin/MaLDyVS#VoxelGrid)
      - [VK-SITS](https://github.com/LaureAcin/MaLDyVS#VK-SITS)
        - [AsyncVKSITS](https://github.com/LaureAcin/MaLDyVS#VK-SITS)
        - [VKSITS](https://github.com/LaureAcin/MaLDyVS#VK-SITS)

## Installation
1. [Python3.8](https://www.python.org/downloads/)  or highter
2. [PyTorch](https://pytorch.org/get-started/locally/) and torchvision packages
3. Few more python packages from [setup.py](https://github.com/LaureAcin/MaLDyVS) file

### Install from the repository
3. git clone https://github.com/LaureAcin/MaLDyVS.git
4. cd mldvs
5. python setup.py install 

## Documentation Generation
To generate the documentation, few packages from sphinx are required:
```bash
pip install myst-parser>=0.15
pip install sphinx-rtd-theme>=1.0
pip install sphinx-autodoc-typehints>=1.12
pip install nbsphinx>=0.8
pip install nbsphinx-link>=1.3
```
Into "/docs" run the following command
```bash
make html
```
in order to generate the documentation in HTML format. Documentation is then available in "build/html",
and the main page is available in "index.html".

## Few Examples
### Event visualisation on DVS128 Gesture dataset

- See [examples/notebooks/event_visualisation/dvs128GestureV2_event_visualisation.ipynb](https://github.com/LaureAcin/MaLDyVS/examples/notebooks/event_visualisation/dvs128GestureV2_event_visualisation.ipynb) to compute available event visualisations on jupyter notebook.
- Examples on other datasets using a notebook are available in [examples/notebooks/event_visualisation/](https://github.com/LaureAcin/MaLDyVS/examples/notebooks/event_visualisation/)

- See [examples/script/event_visualisation/dvs128GestureV2_event_visualisation.py](https://github.com/LaureAcin/MaLDyVS/examples/scripts/event_visualisation/dvs128GestureV2_event_visualisation.py) to compute available event visualisations using command line. 
```
$ python examples/script/dvs128GestureV2_event_visualisation.py
```
It will result in 3 windows, one for each representation. 
- All notebook examples are available using line command in [examples/scripts/]

### Event-based recognition using DVS128-Gesture dataset and VK-SITS event representation
- See [examples/notebooks/event_representation/VK-SITS_classification.ipynb](https://github.com/LaureAcin/MaLDyVS/examples/notebooks/event_representation/VK-SITS_classification.ipynb) to compute available event visualisations on jupyter notebook.
- Examples on other datasets using a notebook are available in [examples/notebooks/event_visualisation/](https://github.com/LaureAcin/MaLDyVS/examples/notebooks/event_visualisation/)

- See [examples/script/event_representation/VK-SITS_classification.py](https://github.com/LaureAcin/MaLDyVS/examples/scripts/event_representation/VK-SITS_classification.py) to compute available event visualisations using command line. 
```
$ python examples/script/VK-SITS_classification.py
> Epoch 1/100
 1/29 █                             - ETA: 4:01 | loss=2.8377 | Top-1 Acc= 7.2 | Top-3 Acc= 34.1 | Top-5 Acc= 52.5
```
- All notebook examples are available using line command in [examples/scripts/]

## Evaluation

Results were obtained in Acin, L., Jacob, P., Simon-Chane, C. and Histace, A.,  VK-SITS: Variable Kernel Speed Invariant Time Surface for Event-Based Recognition.
In Proceedings of the 18th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2023)

[Dataset](https://github.com/LaureAcin/MaLDyVS/src/datawrappers) | [Method](https://github.com/LaureAcin/MaLDyVS/src/mldvs/nn/features/time_surface.py) | Optimizer | Testing Avg. Top-1 Accuracy
--- |---| --- | ---
DVS128 Gesture | VK-SITS | SGD | 87,43
DVS128 Gesture | SITS | SGD | 87,67
DVS128 Gesture | TORE | SGD | 88,16 
DVS128 Gesture | VoxelGrid | SGD | 80,38
DVS128 Gesture | VK-SITS | Adam | 87,92
DVS128 Gesture | SITS | Adam | 88,30
DVS128 Gesture | TORE | Adam | 88,37
DVS128 Gesture | VoxelGrid | Adam | 83,16
N-Caltech101 | VK-SITS | SGD | 73,52
N-Caltech101 | SITS | SGD | 72,66
N-Caltech101 | TORE | SGD | 72,27
N-Caltech101 | VoxelGrid | SGD | 72.29
N-Caltech101 | VK-SITS | Adam | 71,08
N-Caltech101 | SITS | Adam | 71,33
N-Caltech101 | TORE | Adam | 72,21
N-Caltech101 | VoxelGrid | Adam | 72,69

        
##Documentation

###Available datasets
See [/src/mldvs/datawrappers](https://github.com/LaureAcin/MaLDyVS/src/datawrappers) to find all datasets available. 
If you want to add a new dataset, this is here that you will add your dataset wrapper.
You can download an available dataset directly using our package. The tutorial is available in all other tutorials.

- [Cifar10 DVS](https://paperswithcode.com/dataset/cifar10-dvs) consists in a 10 classes datasets: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. It displays images of Cifar10 dataset on a computer display and the event-based cameras moves in front of the display.
- [DVS128 Gesture](https://research.ibm.com/interactive/dvsgesture/) consists in a 11 hand and arm gestures. It has been recorded by subjects which move in a laboratory in front of the event-based camera.
- [N-Caltech101](https://www.garrickorchard.com/datasets/n-caltech101) consists in a 100 objects classes and 1 background class. It displays images of Caltech101 dataset on a computer display and the event-based cameras moves in front of the display.
- [SL-Animals](http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database) consists in a 19 spanish sign language gestures of animals performed in front of an event-based camera in a laboratory. 
This dataset does not contain the train/test split, we need to re-formate it before using the dataset. Look at the [SL-Animals tutorial](https://github.com/LaureAcin/MaLDyVS/examples/notebooks/event_visualisation/SL-Animals_event_visualisation.ipynb). 

###Available event-visualisation methods

Creates a video representation of events by frames of time_window microseconds

- `disp_video(events, time_window, frame_window=1000, frame_duration=(800,800))`

Creates video representation of events by frames of time_window seconds with exponential decay of gray pixel
- `disp_decay_video(events, time_window, exp_decay, frame_duration=1000, window_size=(800,800))`

3D visualization of events
- `disp3D(events, points_size=5)`

####Event representations
##### EST tensor:
Ana I. Maqueda et al. "Event-based vision meets deep learning on steering prediction for self-driving cars". CVPR 2018

Event Counting layer to transform a sequence of events into a tensor.
- `EventCounting(tensor_width, tensor_height)`

#####HOTS: 
Xavier Lagorce et al. "HOTS: a hierarchy of event-based time-surfaces for pattern recognition". TPAMI 2017

A Pytorch Module which compute Time Surfaces for a sequence of events in an asynchronous way.
- `AsyncTimeSurface(radius, decay, tensor_width, tensor_height)`

A Pytorch Module which compute Time Surfaces for a sequence of events. It is a simplified version of the asynchronous: it only takes a batch of event sequences, and return their corresponding Time Surfaces.
- `TimeSurface(radius, decay, tensor_width, tensor_height, n_polarity=2)`

#####HATS:
A. Sironi et al. "HATS: Histograms of averaged time surfaces for robust event-based object classification". CVPR 2018

A Pytorch Module which compute Local Memory Time Surfaces for a sequence of events in an asynchronous way. Implementation follows the idea from HATS paper, but exploit a FIFO implementation as proposed in TORE for better GPU support
- `AsyncLocalMemoryTimeSurface(radius, decay, tensor_width, tensor_height)` 

A Pytorch Module which compute Local Memory Time Surfaces for a sequence of events. Implementation follows the idea from HATS paper, but exploit a FIFO implementation as proposed in TORE for better GPU support.
- `LocalMemoryTimeSurface(radius, decay, tensor_width, tensor_height, memory_length, n_polarity=2)`

#####HATS using FIFO implementation:
A. Sironi et al. "HATS: Histograms of averaged time surfaces for robust event-based object classification".
    CVPR 2018

R. Baldwin et al. "Time-Ordered Recent Event (TORE) Volumes for Event Cameras". ArXiv preprint 2021


A Pytorch module which compute Histogram of Averaged Time Surfaces (HATS) for a sequence of events in an asynchronous way. Implementation follows the idea from HATS paper, but exploit a FIFO implementation as proposed in TORE for better GPU support.
- `AsyncFifoHATS(n_cells, radius, decay, tensor_width, tensor_height, epsilon=1e-6)`

A Pytorch Module which compute Histogram of Averaged Time Surfaces (HATS) for a sequence of events. Implementation follows the idea from HATS paper, but exploit a FIFO implementation as proposed in TORE for better GPU support.
- `FifoHATS(n_cells, radius, decay, tensor_width, tensor_height, memory_length, n_polarity=2, epsilon=1e-6)`

##### TORE
R. Baldwin et al. "Time-Ordered Recent Event (TORE) Volumes for Event Cameras". ArXiv preprint 2021

A Pytorch Module which compute Time-Ordered Recent Events (TORE) for a sequence of events in an asynchronous way.
- `AsyncTORE(min_timestamp=150, max_timestamp=5e6, tensor_width, tensor_height)`

A Pytorch Module which compute Time-Ordered Recent Events (TORE) for a sequence of events.
- `TORE(memory_length, min_timestamp=150, max_timestamp=5e6, n_polarity=2, tensor_width, tensor_height)`

##### SITS
J. Manderscheid et al. "Speed invariant time surface for learning to detect corner points with event-based cameras." CVPR 2019

A Pytorch Module which compute Speed Invariant Time Surface (SITS) for a sequence of events in an asynchronous way.
- `AsyncSITS(radius, tensor_width, tensor_height)`

A Pytorch Module which compute Speed Invariant Time Surface (SITS) for a sequence of events.
- `SITS(radius, tensor_width, tensor_height, n_polarity=2)`

##### VoxelGrid
A. Z. Zhu et al. "Unsupervised event-based learning of optical flow, depth, and egomotion". CVPR 2019

A Pytorch Module which compute Voxel grid for a sequence of events.
- `VoxelGrid(n_polarity=2, n_bins, tensor_width, tensor_height)`

##### VK-SITS
Acin, L., Jacob, P., Simon-Chane, C. and Histace, A., VK-SITS: Variable Kernel Speed Invariant Time Surface for Event-Based Recognition. In Proceedings of the 18th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2023)

A Pytorch Module which compute Variable Kernel Speed Invariant Time Surface (VK-SITS) for a sequence of events in an asynchronous way.
- `AsyncVKSITS(radius, n_filter, tensor_width, tensor_height)`

A Pytorch Module which compute Variable Kernel Speed Invariant Time Surface (VK-SITS) for a sequence of events.
- `VKSITS(radius, n_filter, tensor_width, tensor_height)`
