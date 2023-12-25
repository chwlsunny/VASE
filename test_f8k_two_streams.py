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




# no-finetune(stage one), no mlp, VASE(2-stream)
# vgg16/Davenet
# base
#evaluation_f8k.evalrank_f8k("runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base/model_best.pth.tar", data_path='data/', split="test", fold5=False)
# base+cc, VASE(2-stream)
evaluation_f8k.evalrank_f8k("runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth6.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth12.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)

# resnet50/ResDavenet
# base
#evaluation_f8k.evalrank_f8k("runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base/model_best.pth.tar", data_path='data/', split="test", fold5=False)
# base+cc, VASE(2-stream)
#evaluation_f8k.evalrank_f8k("runs/f8k_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth6.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)




# finetune(stage two), no mlp, VASE(2-stream)
# vgg16/Davenet
# base
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base/model_best.pth.tar", data_path='data/', split="test", fold5=False)
# base+cc, VASE(2-stream)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth6.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_vgg16_vse_base_2rec_l2norm_smooth12.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)

# resnet50/ResDavenet
# base
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base/model_best.pth.tar", data_path='data/', split="test", fold5=False)
# base+cc, VASE(2-stream)
#evaluation_f8k.evalrank_f8k("runs/f8k_finetune_two_streams_bs_32_Bi_GRU_nol2norm2_resnet50_vse_base_2rec_l2norm_smooth6.0/model_best.pth.tar", data_path='data/', split="test", fold5=False)
