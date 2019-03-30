This is the official implementation of "Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation".

For more details, please refer to our [paper](https://arxiv.org/abs/1802.08948). 

### Citing the paper

Please cite the paper in your publications if it helps your research:
```
@inproceedings{lyu2018multi,
      title={Multi-oriented scene text detection via corner localization and region segmentation},
      author={Lyu, Pengyuan and Yao, Cong and Wu, Wenhao and Yan, Shuicheng and Bai, Xiang},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={7553--7563},
      year={2018}
}
``` 
### Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Models](#models)
4. [Test](#test)
5. [Train](#train)
6. [License](#license)

### Requirements
- NVIDIA GPU, Ubuntu 14.04, Python2.7, CUDA8/9
- PyTorch 0.2.0_3

### Installation

```
git clone https://github.com/lvpengyuan/corner.git
sh ./make.sh   or  cd rpsroi_pooling && python build.py
```

### Models
Download the model and place it in ```weights/```

Our trained model:
[Google Drive](https://drive.google.com/open?id=159kPFUtFddvRxQqMm4ewv8UIZ91-Y8oT);

### Test

You can test a model in a single scale:
```
python eval_all.py
```
or in multi-scale: 
```
python eval_multiscale.py
```
Note that, you should modify the model path and the test dataset before testing. 

### Train
```
python train.py
```
To train a new model, you should modify the training settings before training.


### License
This code is only for academic purpose.

