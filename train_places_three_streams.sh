#!/usr/bin/env bash

# no-finetune(stage one), no mlp, VASE
# vgg16/DAVEnet
# base+tri
#CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 35 --batch_size 32 --num_streams 3 --learning_rate 0.0002 --cnn_type vgg16
# base+tri+cc, VASE
CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0 --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 35 --batch_size 32 --num_streams 3 --learning_rate 0.0002 --cnn_type vgg16

# resnet50/ResDAVEnet
# base+tri
#CUDA_VISIBLE_DEVICES=4 python train_places.py --data_path data/ --data_name places --logger_name runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 35 --batch_size 32 --num_streams 3 --learning_rate 0.0002 --cnn_type resnet50
# base+tri+cc, VASE
#CUDA_VISIBLE_DEVICES=4 python train_places.py --data_path data/ --data_name places --logger_name runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth4.0 --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 35 --batch_size 32 --num_streams 3 --learning_rate 0.0002 --cnn_type resnet50




# finetune(stage two), no mlp, VASE
# vgg16/DAVEnet
# base+tri
#CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base --resume runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base/model_best.pth.tar --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 40 --batch_size 32 --num_streams 3 --finetune --learning_rate 0.00002 --cnn_type vgg16
# base+tri+cc, VASE
#CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0 --resume runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0/model_best.pth.tar --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 40 --batch_size 32 --num_streams 3 --finetune --learning_rate 0.00002 --cnn_type vgg16

# resnet50/ResDAVEnet
# base+tri
#CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base --resume runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base/model_best.pth.tar --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 40 --batch_size 32 --num_streams 3 --finetune --learning_rate 0.00002 --cnn_type resnet50
# base+tri+cc, VASE
#CUDA_VISIBLE_DEVICES=6 python train_places.py --data_path data/ --data_name places --logger_name runs/places_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth4.0 --resume runs/places_three_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth4.0/model_best.pth.tar --data_train data/PlacesAudio_400k_distro/metadata/train.json --data_val data/PlacesAudio_400k_distro/metadata/val.json --num_epoch 40 --batch_size 32 --num_streams 3 --finetune --learning_rate 0.00002 --cnn_type resnet50
