import torch
import numpy as np
import random
import pickle
import os
import time
import shutil
import data
from vocab import Vocabulary  # NOQA
from model import VSE
from evaluation_f8k import i2a, a2i, AverageMeter, LogCollector, encode_data

import logging
#import librosa
import tensorboard_logger as tb_logger
import argparse
import pdb

# add for fixing random seed
#GLOBAL_SEED=1024
#GLOBAL_SEED=100
GLOBAL_SEED=1234567890

def set_seed(seed):
    random.seed(seed) # set random seed for python
    np.random.seed(seed) # set random seed for numpy
    torch.manual_seed(seed) # set random seed for cpu
    torch.cuda.manual_seed(seed) # set random seed for gpu 
    torch.cuda.manual_seed_all(seed) # set random seed for all available gpus
    os.environ['PYTHONHASHSEED']=str(seed) # set random seed for hash
    torch.backends.cudnn.benchmark=False        # do not use fuzzy algorithms of cudnn
    torch.backends.cudnn.deterministic=True
    #torch.backends.cuddn.enabled=False

set_seed(GLOBAL_SEED)


#GLOBAL_WORKER_ID=None
def worker_init_fn(worker_id):
    #global GLOBAL_WOKER_ID
    #GLOBAL_WORKER_ID=worker_id
    set_seed(GLOBAL_SEED+worker_id)


#torch.set_num_threads(4)


#os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.environ['CUDA_VISIBLE_DEVICES']='1'
#os.environ['CUDA_VISIBLE_DEVICES']='2'
#os.environ['CUDA_VISIBLE_DEVICES']='3'
#os.environ['CUDA_VISIBLE_DEVICES']='4'
#os.environ['CUDA_VISIBLE_DEVICES']='5'
#os.environ['CUDA_VISIBLE_DEVICES']='6'
#os.environ['CUDA_VISIBLE_DEVICES']='7'
#os.environ['CUDA_VISIBLE_DEVICES']='7,6'

torch.set_num_threads(2)

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/chengwenlong/code/vsepp/data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='coco_precomp',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k|places')
    parser.add_argument('--data_train', type=str, default='',
                        help='training data json')
    parser.add_argument('--data_val', type=str, default='',
                        help='validation data json')
    parser.add_argument('--vocab_path', default='/root/wenlong_workspace/code/vsepp_a_t_i/vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--margin_3s', default=0.09, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--num_streams', default=2, type=int,
                        help='The number of the streams of the model..')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--use_third_order_loss', action='store_true',
                        help='Use third-order loss.')
    parser.add_argument('--lambda1', default=1.0, type=float,
                        help='The balance parameter, second-order information interactive loss + lambda1 * third-order information interactive loss.')
    parser.add_argument('--cnn_type', default='vgg19',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    #parser.add_argument('--reset_train', action='store_true',
    #                    help='Ensure the training is always done in '
    #                    'train mode (Not recommended).')
    parser.add_argument('--reset_train', default=True,
                        help='Ensure the training is always done in '
                        'train mode (Not recommended).')
    opt = parser.parse_args()
    opt.worker_init_fn=worker_init_fn
    print(opt)

    #pdb.set_trace()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)
        
    # Load data loaders
    if opt.data_name=='f8k':
        train_loader, val_loader = data.get_loaders_two_three_streams(
            opt.data_name, vocab, opt.crop_size, opt.batch_size, opt.workers, opt)
    else:
        #raise ValueError('data_name should be %s or %s' %('f8k','places'))
        raise ValueError('data_name should be %s' %('f8k'))


    #pdb.set_trace()

    # Construct the model
    model=VSE(opt)


    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            #pdb.set_trace()
            #checkpoint['opt']['worker_init_fn']=None
            checkpoint['opt'].worker_init_fn=None
            torch.save(checkpoint,opt.resume)
            model.load_state_dict(checkpoint['model'])
            #model.load_state_dict_two_streams(checkpoint['model'])

            # validation
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))


    #pdb.set_trace()

    # Train the Model
    best_rsum = 0
    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)
        #pdb.set_trace()
        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        #pdb.set_trace()
        # evaluate on validation set
        rsum = validate(opt, val_loader, model)

        #pdb.set_trace()
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        #pdb.set_trace()
        opt.worker_init_fn=None
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            #'model': model.state_dict_two_streams(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    #pdb.set_trace()
    # switch to train mode

    #if torch.cuda.is_available():
    #    model.cuda()

    model.train_start()

    #pdb.set_trace()

    end = time.time()
    for i, train_data in enumerate(train_loader):

        if opt.reset_train:
            # Always reset to train mode, this is not the default behavior
            model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions

    if opt.num_streams==2:
        img_embs, audio_embs = encode_data(
            model, val_loader, opt.num_streams, opt.log_step, logging.info)

        # image-to-speech retrieval
        (r1, r5, r10, medr, meanr) = i2a(img_embs, audio_embs, measure=opt.measure)
        logging.info("Image to audio: %.1f, %.1f, %.1f, %.1f, %.1f" %
                     (r1, r5, r10, medr, meanr))
        # speech-to-image retrieval
        (r1i, r5i, r10i, medri, meanri) = a2i(img_embs, audio_embs, measure=opt.measure)
        logging.info("Audio to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                     (r1i, r5i, r10i, medri, meanri))
        # sum of recalls to be used for early stopping
        currscore = r1 + r5 + r10 + r1i + r5i + r10i

        # record metrics in tensorboard
        tb_logger.log_value('r1', r1, step=model.Eiters)
        tb_logger.log_value('r5', r5, step=model.Eiters)
        tb_logger.log_value('r10', r10, step=model.Eiters)
        tb_logger.log_value('medr', medr, step=model.Eiters)
        tb_logger.log_value('meanr', meanr, step=model.Eiters)
        tb_logger.log_value('r1i', r1i, step=model.Eiters)
        tb_logger.log_value('r5i', r5i, step=model.Eiters)
        tb_logger.log_value('r10i', r10i, step=model.Eiters)
        tb_logger.log_value('medri', medri, step=model.Eiters)
        tb_logger.log_value('meanri', meanri, step=model.Eiters)
        tb_logger.log_value('rsum', currscore, step=model.Eiters)

        return currscore

    elif opt.num_streams==3:
        img_embs, cap_embs, audio_embs = encode_data(
            model, val_loader, opt.num_streams, opt.log_step, logging.info)

        # image-to-speech retrieval
        (r1, r5, r10, medr, meanr) = i2a(img_embs, audio_embs, measure=opt.measure)
        logging.info("Image to audio: %.1f, %.1f, %.1f, %.1f, %.1f" %
                     (r1, r5, r10, medr, meanr))
        # speech-to-image retrieval
        (r1i, r5i, r10i, medri, meanri) = a2i(img_embs, audio_embs, measure=opt.measure)
        logging.info("Audio to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                     (r1i, r5i, r10i, medri, meanri))
        # sum of recalls to be used for early stopping
        currscore = r1 + r5 + r10 + r1i + r5i + r10i

        # record metrics in tensorboard
        tb_logger.log_value('r1', r1, step=model.Eiters)
        tb_logger.log_value('r5', r5, step=model.Eiters)
        tb_logger.log_value('r10', r10, step=model.Eiters)
        tb_logger.log_value('medr', medr, step=model.Eiters)
        tb_logger.log_value('meanr', meanr, step=model.Eiters)
        tb_logger.log_value('r1i', r1i, step=model.Eiters)
        tb_logger.log_value('r5i', r5i, step=model.Eiters)
        tb_logger.log_value('r10i', r10i, step=model.Eiters)
        tb_logger.log_value('medri', medri, step=model.Eiters)
        tb_logger.log_value('meanri', meanri, step=model.Eiters)
        tb_logger.log_value('rsum', currscore, step=model.Eiters)

        return currscore

    else:

        raise ValueError('num_streams should be %d or %d' %(2,3))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
