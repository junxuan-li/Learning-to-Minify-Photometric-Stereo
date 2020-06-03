# Learning to Minify Photometric Stereo

 [Learning to Minify Photometric Stereo, CVPR 2019.](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Learning_to_Minify_Photometric_Stereo_CVPR_2019_paper.pdf)


## Dependencies

The code was tested on:
- Ubuntu 16.04
- Python 3.6.6 
- Keras 2.1.3 
- Tensorflow (GPU) 1.2.0
- OpenCV

With GPU: NVIDIA GeForce GTX 1080 Ti.


## Overview

- We provided the trained models on 6, 8, 10, 16 and 96 inputs. You can find them in folder:  `./learned_model`

- We also provided a pre-processed [DiLiGenT](https://sites.google.com/site/photometricstereodata/single?authuser=0) dataset for testing. You can find it in folder: `./testing_data/14`


## Running the tests

To test the models on DiLiGenT main dataset

```
python ps_cnn_test.py
```

If you want to test the model in a different dataset, change the `test_data_path` in `ps_cnn_test.py` to the target dataset. The code will regenerate a pre-processed observation map for the testing later.

## Citation
If you find this code useful in your reasearch, please consider cite:
```
@inproceedings{li2019learning,
  title={Learning to minify photometric stereo},
  author={Li, Junxuan and Robles-Kelly, Antonio and You, Shaodi and Matsushita, Yasuyuki},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7568--7576},
  year={2019}
}
```
