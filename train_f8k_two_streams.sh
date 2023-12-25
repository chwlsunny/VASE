# no-finetune(stage one), no mlp, VASE(2-stream)
# vgg16/Davenet
# base
CUDA_VISIBLE_DEVICES=6 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base --num_epochs 100 --batch_size 32 --num_streams 2 --learning_rate 0.0002 --cnn_type vgg16
# base+cc, VASE(2-stream)
CUDA_VISIBLE_DEVICES=6 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth6.0 --num_epochs 100 --batch_size 32 --num_streams 2 --learning_rate 0.0002 --cnn_type vgg16
#CUDA_VISIBLE_DEVICES=6 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth12.0 --num_epochs 100 --batch_size 32 --num_streams 2 --learning_rate 0.0002 --cnn_type vgg16

# resnet50/ResDAVEnet
# base
#CUDA_VISIBLE_DEVICES=5 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base --num_epochs 100 --batch_size 32 --num_streams 2 --learning_rate 0.0002 --cnn_type resnet50
# base+cc, VASE(2-stream)
#CUDA_VISIBLE_DEVICES=5 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth6.0 --num_epochs 100 --batch_size 32 --num_streams 2 --learning_rate 0.0002 --cnn_type resnet50




# finetune(stage two), no mlp, VASE(2-stream)
# vgg16/DAVEnet
# base
#CUDA_VISIBLE_DEVICES=4 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base --resume runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base/model_best.pth.tar --num_epochs 100 --batch_size 32 --num_streams 2 --finetune --learning_rate 0.00002 --cnn_type vgg16
# base+cc, VASE(2-stream)
#CUDA_VISIBLE_DEVICES=4 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth6.0 --resume runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth6.0/model_best.pth.tar --num_epochs 100 --batch_size 32 --num_streams 2 --finetune --learning_rate 0.00002 --cnn_type vgg16
#CUDA_VISIBLE_DEVICES=4 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth12.0 --resume runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth12.0/model_best.pth.tar --num_epochs 100 --batch_size 32 --num_streams 2 --finetune --learning_rate 0.00002 --cnn_type vgg16

# resnet50/ResDAVEnet
# base
#CUDA_VISIBLE_DEVICES=5 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base --resume runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base/model_best.pth.tar --num_epochs 100 --batch_size 32 --num_streams 2 --finetune --learning_rate 0.00002 --cnn_type resnet50
# base+cc, VASE(2-stream)
#CUDA_VISIBLE_DEVICES=5 python train_f8k.py --data_path data/ --data_name f8k --logger_name runs/f8k_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth6.0 --resume runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth6.0/model_best.pth.tar --num_epochs 100 --batch_size 32 --num_streams 2 --finetune --learning_rate 0.00002 --cnn_type resnet50
