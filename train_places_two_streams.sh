#!/usr/bin/env bash

# no-finetune(stage one), no mlp, VASE(2-stream)
# vgg16/DAVEnet
# base
#CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 35 --batch_size 32 --num_streams 2 --learning_rate 0.0002 --cnn_type vgg16
# base+cc, VASE(2-stream)
CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth12.0 --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 35 --batch_size 32 --num_streams 2 --learning_rate 0.0002 --cnn_type vgg16

# resnet50/ResDAVEnet
# base
#CUDA_VISIBLE_DEVICES=4 python train_places.py --data_path data/ --data_name places --logger_name runs/places_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 35 --batch_size 32 --num_streams 2 --learning_rate 0.0002 --cnn_type resnet50
# base+cc, VASE(2-stream)
#CUDA_VISIBLE_DEVICES=4 python train_places.py --data_path data/ --data_name places --logger_name runs/places_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth6.0 --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 35 --batch_size 32 --num_streams 2 --learning_rate 0.0002 --cnn_type resnet50




# finetune(stage two), no mlp, VASE(2-stream)
# vgg16/DAVEnet
# base
#CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base --resume runs/places_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base/model_best.pth.tar --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 40 --batch_size 32 --num_streams 2 --finetune --learning_rate 0.00002 --cnn_type vgg16
# base+cc, VASE(2-stream)
#CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth12.0 --resume runs/places_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth12.0/model_best.pth.tar --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 40 --batch_size 32 --num_streams 2 --finetune --learning_rate 0.00002 --cnn_type vgg16

# resnet50/ResDAVEnet
# base
#CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base --resume runs/places_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base/model_best.pth.tar --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 40 --batch_size 32 --num_streams 2 --finetune --learning_rate 0.00002 --cnn_type resnet50
# base+cc, VASE(2-stream)
#CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth6.0 --resume runs/places_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth6.0/model_best.pth.tar --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 40 --batch_size 32 --num_streams 2 --finetune --learning_rate 0.00002 --cnn_type resnet50
