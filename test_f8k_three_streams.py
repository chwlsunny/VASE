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




# no-finetune(stage one), no mlp, VASE
# vgg16/Davenet
# base+tri
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base/model_best.pth.tar", data_path='data/', split="test", fold5=False)
# base+tri+cc, VASE
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth0.8/model_best.pth.tar", data_path='data/', split="test", fold5=False)
evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth1.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth1.5/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth4.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth4.5/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.2/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.3/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.4/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.5/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.6/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.7/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth6.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth8.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth9.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth10.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth12.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)

# resnet50/ResDavenet
# base+tri
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base/model_best.pth.tar", data_path='data/', split="test", fold5=False)
# base+tri+cc, VASE
#evaluation_f8k.evalrank_f8k("runs/f8k_three_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth6.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)



# finetune(stage two), no mlp, VASE
# vgg16/Davenet
# base+tri
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base/model_best.pth.tar", data_path='data/', split="test", fold5=False)
# base+tri+cc, VASE
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth0.8/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth1.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth1.5/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth2.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth4.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth4.5/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.2/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.3/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.4/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.5/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.6/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth5.7/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth6.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth8.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth9.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth10.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth12.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)

# resnet50/ResDavenet
# base+tri
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base/model_best.pth.tar", data_path='data/', split="test", fold5=False)
# base+tri+cc, VASE
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_three_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth6.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
