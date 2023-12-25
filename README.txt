

Our code is based on https://github.com/fartashf/vsepp, https://github.com/kuanghuei/SCAN, https://github.com/dharwath/DAVEnet-pytorch and https://github.com/wnhsu/ResDAVEnet-VQ.


Dependencies
We recommended to use Anaconda for the following packages.
1. Python 2.7
2. Pytorch (>0.4.0)
3. Numpy (>1.12.1)
4. librosa
5. TensorBoard
6. pycocotools
7. torchvision
8. matplotlib
9. Punkt Sentence Tokenizer:
    import nltk
    nltk.download()
    > d punkt


Download data （在data文件夹下“数据集所在的位置.txt”中记录了相关数据在dgx-1服务器中的位置）
1. Flickr8K dataset
    1)https://forms.illinois.edu/sec/1713398 (image and text)
    2)https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads.cgi (speech)
2. Places dataset
    1)http://places.csail.mit.edu/ (image)
    2)http://groups.csail.mit.edu/sls/downloads/placesaudio/downloads.cgi (speech and text)


Training new models on Flickr8K dataset (Refer to train_f8k_two_streams.sh and train_f8k_three_streams.sh)
# stage one
CUDA_VISIBLE_DEVICES=5 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth1.0 --num_epochs 100 --batch_size 32 --num_streams 3 --learning_rate 0.0002 --cnn_type vgg16
# stage two
CUDA_VISIBLE_DEVICES=5 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth1.0 --resume runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth1.0/model_best.pth.tar --num_epochs 100 --batch_size 32 --num_streams 3 --finetune --learning_rate 0.00002 --cnn_type vgg16

Training new models on Places dataset (Refer to train_places_two_streams.sh and train_places_three_streams.sh)
# stage one
CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0 --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 35 --batch_size 32 --num_streams 3 --learning_rate 0.0002 --cnn_type vgg16
# stage two
CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0 --resume runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0/model_best.pth.tar --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 40 --batch_size 32 --num_streams 3 --finetune --learning_rate 0.00002 --cnn_type vgg16


Evaluation on Flickr8K dataset (Refer to test_f8k_two_stream.sh and test_f8k_three_streams.sh)
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


