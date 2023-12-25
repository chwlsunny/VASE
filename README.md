# Improving Visual-Semantic Embeddings with Hard Negatives

Our code is based on https://github.com/fartashf/vsepp, https://github.com/kuanghuei/SCAN, https://github.com/dharwath/DAVEnet-pytorch and https://github.com/wnhsu/ResDAVEnet-VQ.

Code for the image-speech retrieval methods from
**[A Reconstruction-based Visual-Acoustic-Semantic Embedding Method for Speech-Image Retrieval](https://ieeexplore.ieee.org/document/9765364)**
* Cheng W, Tang W, Huang Y, Luo Y, Wang L, IEEE Transactions on Multimedia (TMM), 2022.

## Dependencies
We recommended to use Anaconda for the following packages.

* Python 2.7
* [PyTorch](http://pytorch.org/) (>0.4.0)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [librosa]
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* [pycocotools](https://github.com/cocodataset/cocoapi)
* [torchvision]
* [matplotlib]

* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data

Download data
1. Flickr8K dataset
    1)https://forms.illinois.edu/sec/1713398 (image and text)
    2)https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads.cgi (speech)
2. Places dataset
    1)http://places.csail.mit.edu/ (image)
    2)http://groups.csail.mit.edu/sls/downloads/placesaudio/downloads.cgi (speech and text)

## Train models

### Training new models on Flickr8K dataset (Refer to train_f8k_two_streams.sh and train_f8k_three_streams.sh)

#### stage one
```shell
CUDA_VISIBLE_DEVICES=5 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth1.0 --num_epochs 100 --batch_size 32 --num_streams 3 --learning_rate 0.0002 --cnn_type vgg16
```

#### stage two
```shell
CUDA_VISIBLE_DEVICES=5 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth1.0 --resume runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth1.0/model_best.pth.tar --num_epochs 100 --batch_size 32 --num_streams 3 --finetune --learning_rate 0.00002 --cnn_type vgg16
```

### Training new models on Places dataset (Refer to train_places_two_streams.sh and train_places_three_streams.sh)

#### stage one
```shell
CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0 --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 35 --batch_size 32 --num_streams 3 --learning_rate 0.0002 --cnn_type vgg16
```

#### stage two
```shell
CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0 --resume runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0/model_best.pth.tar --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 40 --batch_size 32 --num_streams 3 --finetune --learning_rate 0.00002 --cnn_type vgg16
```



## Evaluate on Flickr8K dataset (Refer to test_f8k_two_stream.sh and test_f8k_three_streams.sh)

```python
import torch
from vocab import Vocabulary
import evaluation_f8k
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
```

## Reference

If you found this code useful, please cite the following paper:

    @article{Cheng2023ARV,
      title={A Reconstruction-Based Visual-Acoustic-Semantic Embedding Method for Speech-Image Retrieval},
      author={Wenlong Cheng and Wei Tang and Yan Huang and Yiwen Luo and Liang Wang},
      journal = {IEEE Transactions on Multimedia},
      year={2023},
      volume={25},
      url = {https://github.com/chwlsunny/VASE}
    }
