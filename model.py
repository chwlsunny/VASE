import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
#from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from models.AudioModels import ResDavenet
import pdb

class Davenet_rnn(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(Davenet_rnn, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        #self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1)) #### version_1
        self.pool = nn.AvgPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1)) #### version_2

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x) ####
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.squeeze(2)
        return x

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Davenet_rnn, self).load_state_dict(new_state)

def Encoderaudio(data_name, embed_size):
    if data_name.endswith('_precomp'):
        pass                                    ####
    else:
        # choose speech encoder manually
        #aud_enc = ResDavenet(embed_size)  #ResDAVEnet
        aud_enc = Davenet_rnn(embed_size) # DAVEnet

    return aud_enc


#def l2norm(X, eps=1e-8):
def l2norm(X, dim=1, eps=1e-8):
    """L2-normalize columns of X
    """
    #norm = (torch.pow(X, 2).sum(dim=1, keepdim=True)+eps).sqrt()     # previous version
    norm = (torch.pow(X, 2).sum(dim=dim, keepdim=True)).sqrt()+eps    # revise version
    X = torch.div(X, norm)
    return X


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('_precomp'):
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, use_abs, no_imgnorm)
    else:
        img_enc = EncoderImageFull(
            embed_size, finetune, cnn_type, use_abs, no_imgnorm)

    return img_enc


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        #pdb.set_trace()

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderText, self).load_state_dict(new_state)

# RNN Based Speech Model
class AudioRNN(nn.Module):

    def __init__(self, audio_size, hidden_size, num_layers,
                 use_abs=False):
        super(AudioRNN, self).__init__()
        self.use_abs = use_abs
        self.embed_size = hidden_size

        
        # speech GRU
        #self.rnn = nn.GRU(audio_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.GRU(audio_size, hidden_size, num_layers, batch_first=True, bidirectional=True) #### bidirectional code v2
        #self.rnn = nn.LSTM(audio_size, hidden_size, num_layers, batch_first=True, bidirectional=True) #### bidirectional code v2
        #self.classifier = nn.Linear(hidden_size*2, hidden_size) #### bidirectional code v2
        #self.rnn = nn.GRU(audio_size, hidden_size/2, num_layers, batch_first=True, bidirectional=True) #### bidirectional code v1

    def forward(self, x, lengths):
        """Handles variable size captions
        """

        x=l2norm(x) # audio normalization

        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        #pdb.set_trace()
        #out = torch.gather(padded[0], 1, I).squeeze(1)
        #out = torch.gather((padded[0][:,:,:padded[0].size(2)/2]+padded[0][:,:,padded[0].size(2)/2:])/2, 1, I).squeeze(1)
        out = torch.gather((padded[0][:,:,:padded[0].size(2)//2]+padded[0][:,:,padded[0].size(2)//2:])/2, 1, I).squeeze(1)
        #out = torch.gather(self.classifier(padded[0]), 1, I).squeeze(1) #### code v2

        # normalization in the joint embedding space
        #out = l2norm(out) #flickr8k normalization, places no normalization

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(AudioRNN, self).load_state_dict(new_state)


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class MLP3(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(MLP3,self).__init__()
        self.fc1=nn.Linear(input_size,hidden_size) # layer 1
        self.fc2=nn.Linear(hidden_size,hidden_size) # layer 2
        self.fc3=nn.Linear(hidden_size,output_size) # layer 3
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        return x

    def load_state_dict(self,state_dict):
        """
        Copies parameters, overwritting the default one to
        accept state_dict from Full model
        """
        own_state=self.state_dict()
        new_state=OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name]=param

        super(MLP3, self).load_state_dict(new_state)


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        #self.margin1 = 0.12
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-speech score matrix
        scores = self.sim(im, s)

        diagonal = scores.diag().view(im.size(0), 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # speech retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)


        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        #pdb.set_trace()
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        #pdb.set_trace()

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        cost_base=cost_s.sum()+cost_im.sum()

        return cost_base                # base triplet loss

# feature reconstruction between two data modalities
class re_feats(nn.Module):
    """
    Reconstruct features
    """

    def __init__(self, smooth=6.0):
        super(re_feats, self).__init__()
        self.smooth=smooth
        self.sim=cosine_sim

    #def forward(self, im ,s):
    def forward(self, im ,aud):
        scores=self.sim(im, aud)
        scores_1=nn.LeakyReLU(0.1)(scores)
        scores_2=nn.LeakyReLU(0.1)(scores).t()

        scores_1=l2norm(scores_1, 1)
        scores_1=nn.Softmax(dim=-1)(scores_1*self.smooth)
        scores_2=l2norm(scores_2, 1)
        scores_2=nn.Softmax(dim=-1)(scores_2*self.smooth)

        im_rec=scores_1.mm(aud)
        aud_rec=scores_2.mm(im)

        #return im_rec, s_rec
        return im_rec, aud_rec


# feature reconstruction among three modalities
class re_feats_three_streams(nn.Module):                 # v2
    """
    Reconstruct features
    """

    def __init__(self, smooth=6.0):
        super(re_feats_three_streams, self).__init__()
        self.smooth=smooth
        self.sim=cosine_sim

    def forward(self, im , aud, cap):

        # the reconstruction of text features
        im_aud=torch.cat((im, aud), dim=0)
        scores=self.sim(im_aud, cap)
        scores=nn.LeakyReLU(0.1)(scores)

        scores=l2norm(scores, 1)
        scores=nn.Softmax(dim=-1)(scores*self.smooth)

        scores=scores.t()
        cap_rec=scores.mm(im_aud)

        # the reconstruction of speech features
        im_cap=torch.cat((im, cap), dim=0)
        scores1=self.sim(im_cap, aud)
        scores1=nn.LeakyReLU(0.1)(scores1)        

        scores1=l2norm(scores1, 1)
        scores1=nn.Softmax(dim=-1)(scores1*self.smooth)

        scores1=scores1.t()
        aud_rec=scores1.mm(im_cap)

        # the reconstruction of image features
        aud_cap=torch.cat((aud, cap), dim=0)
        scores2=self.sim(aud_cap, im)
        scores2=nn.LeakyReLU(0.1)(scores2)

        scores2=l2norm(scores2, 1)
        scores2=nn.Softmax(dim=-1)(scores2*self.smooth)

        scores2=scores2.t()
        im_rec=scores2.mm(aud_cap)

        return im_rec, aud_rec, cap_rec


class VSE(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        self.grad_clip = opt.grad_clip
        self.num_streams = opt.num_streams
        #self.re_feats=re_feats(smooth=6.0)
        #self.re_feats=re_feats_three_streams(smooth=6.0)
        if opt.num_streams==2:
            self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                        opt.finetune, opt.cnn_type,
                                        use_abs=opt.use_abs,
                                        no_imgnorm=opt.no_imgnorm)
            self.aud_enc = Encoderaudio(opt.data_name, opt.embed_size)
            self.aud_rnn = AudioRNN(opt.embed_size, opt.embed_size, 
                                           opt.num_layers, use_abs=opt.use_abs) ####
            #self.img_rec=MLP3(opt.embed_size, opt.embed_size, opt.embed_size) # do not share the weights of parameters
            #self.aud_rec=MLP3(opt.embed_size, opt.embed_size, opt.embed_size) # do not share the weights of parameters
            #self.feats_rec=MLP3(opt.embed_size, opt.embed_size, opt.embed_size) # share the weights of parameters
            #self.re_feats=re_feats(smooth=2.0)
            #self.re_feats=re_feats(smooth=4.0)
            self.re_feats=re_feats(smooth=6.0)
            #self.re_feats=re_feats(smooth=7.0)
            #self.re_feats=re_feats(smooth=7.5)
            #self.re_feats=re_feats(smooth=7.8)
            #self.re_feats=re_feats(smooth=7.9)
            #self.re_feats=re_feats(smooth=8.0)
            #self.re_feats=re_feats(smooth=8.02)
            #self.re_feats=re_feats(smooth=8.1)
            #self.re_feats=re_feats(smooth=8.2)
            #self.re_feats=re_feats(smooth=8.5)
            #self.re_feats=re_feats(smooth=9.0)
            #self.re_feats=re_feats(smooth=10.0)
            #self.re_feats=re_feats(smooth=12.0)
            if torch.cuda.is_available():
                self.img_enc.cuda()
                self.aud_enc.cuda()
                self.aud_rnn.cuda()
                #self.img_rec.cuda() # do not share the weights of parameters
                #self.aud_rec.cuda() # do not share the weights of parameters
                #self.feats_rec.cuda() # share the weights of parameters
                #self.aud_enc.cuda() ####
                #cudnn.benchmark = True
        elif opt.num_streams==3:
            self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                        opt.finetune, opt.cnn_type,
                                        use_abs=opt.use_abs,
                                        no_imgnorm=opt.no_imgnorm)
            self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                       opt.embed_size, opt.num_layers,
                                       use_abs=opt.use_abs)
            self.aud_enc = Encoderaudio(opt.data_name, opt.embed_size) ####
            self.aud_rnn = AudioRNN(opt.embed_size, opt.embed_size, 
                                           opt.num_layers, use_abs=opt.use_abs) ####
            #self.img_rec=MLP3(opt.embed_size, opt.embed_size, opt.embed_size) # do not share the weights of parameters
            #self.aud_rec=MLP3(opt.embed_size, opt.embed_size, opt.embed_size) # do not share the weights of parameters
            #self.feats_rec=MLP3(opt.embed_size, opt.embed_size, opt.embed_size) # share the weights of parameters
            #self.re_feats=re_feats_three_streams(smooth=0.8)
            #self.re_feats=re_feats_three_streams(smooth=1.0)
            #self.re_feats=re_feats_three_streams(smooth=1.5)
            #self.re_feats=re_feats_three_streams(smooth=2.0)
            self.re_feats=re_feats_three_streams(smooth=4.0)
            #self.re_feats=re_feats_three_streams(smooth=4.5)
            #self.re_feats=re_feats_three_streams(smooth=5.0)
            #self.re_feats=re_feats_three_streams(smooth=5.2)
            #self.re_feats=re_feats_three_streams(smooth=5.3)
            #self.re_feats=re_feats_three_streams(smooth=5.4)
            #self.re_feats=re_feats_three_streams(smooth=5.5)
            #self.re_feats=re_feats_three_streams(smooth=5.6)
            #self.re_feats=re_feats_three_streams(smooth=5.7)
            #self.re_feats=re_feats_three_streams(smooth=6.0)
            #self.re_feats=re_feats_three_streams(smooth=8.0)
            #self.re_feats=re_feats_three_streams(smooth=9.0)
            #self.re_feats=re_feats_three_streams(smooth=10.0)
            #self.re_feats=re_feats_three_streams(smooth=12.0)
            if torch.cuda.is_available():
                self.img_enc.cuda()
                self.txt_enc.cuda()
                self.aud_enc.cuda() ####
                self.aud_rnn.cuda()
                #self.img_rec.cuda() # do not share the weights of parameters
                #self.aud_rec.cuda() # do not share the weights of parameters
                #self.feats_rec.cuda() # share the weights of parameters
                #cudnn.benchmark = True
        else:
            raise ValueError('num_streams shoud be %d or %d' %(2,3))

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)


        # Three streams, text(caption), image, audio
        if opt.num_streams==3:
            params = list(self.txt_enc.parameters())
            params += list(self.img_enc.fc.parameters())
            params += list(self.aud_enc.parameters()) ####
            params += list(self.aud_rnn.parameters())
            #params += list(self.img_rec.parameters()) # do not share the weights of parameters
            #params += list(self.aud_rec.parameters()) # do not share the weights of parameters
            #params += list(self.feats_rec.parameters()) # share the weights of parameters
        elif opt.num_streams==2:
            # Two streams, image, audio
            #params = list(self.txt_enc.parameters())
            params = list(self.img_enc.fc.parameters())
            params += list(self.aud_enc.parameters()) ####
            params += list(self.aud_rnn.parameters()) ####
            #params += list(self.img_rec.parameters()) # do not share the weights of parameters
            #params += list(self.aud_rec.parameters()) # do not share the weights of parameters
            #params += list(self.feats_rec.parameters()) # share the weights of parameters
        else:
            raise ValueError('num_streams shoud be %d or %d' %(2,3))

        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        if self.num_streams==2:
            #state_dict = [self.img_enc.state_dict(), self.aud_enc.state_dict()]
            state_dict = [self.img_enc.state_dict(), self.aud_enc.state_dict(), self.aud_rnn.state_dict()]
            #state_dict = [self.img_enc.state_dict(), self.aud_enc.state_dict(), self.aud_rnn.state_dict(), self.img_rec.state_dict(), self.aud_rec.state_dict()] # do not share the weights of parameters
            #state_dict = [self.img_enc.state_dict(), self.aud_enc.state_dict(), self.aud_rnn.state_dict(), self.feats_rec.state_dict()] # share the weights of parameters
        elif self.num_streams==3:
            state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.aud_enc.state_dict(), self.aud_rnn.state_dict()]
            #state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.aud_enc.state_dict(), self.aud_rnn.state_dict(), self.img_rec.state_dict(), self.aud_rec.state_dict()] # do not share the weights of parameters
            #state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.aud_enc.state_dict(), self.aud_rnn.state_dict(), self.feats_rec.state_dict()] # share the weights of parameters
        else:
            raise ValueError('num_streams should be %d or %d' %(2,3))	
        return state_dict


    def load_state_dict(self, state_dict):
        if self.num_streams==2:
            self.img_enc.load_state_dict(state_dict[0])
            self.aud_enc.load_state_dict(state_dict[1])
            self.aud_rnn.load_state_dict(state_dict[2]) ####
            #self.img_rec.load_state_dict(state_dict[3]) # do not share the weights of parameters
            #self.aud_rec.load_state_dict(state_dict[4]) # do not share the weights of parameters
            #self.feats_rec.load_state_dict(state_dict[3]) # share the weights of parameters
        elif self.num_streams==3:
            self.img_enc.load_state_dict(state_dict[0])
            self.txt_enc.load_state_dict(state_dict[1])
            self.aud_enc.load_state_dict(state_dict[2]) ####
            self.aud_rnn.load_state_dict(state_dict[3]) ####
            #self.img_rec.load_state_dict(state_dict[4]) # do not share the weights of parameters
            #self.aud_rec.load_state_dict(state_dict[5]) # do not share the weights of parameters
            #self.feats_rec.load_state_dict(state_dict[4]) # share the weights of parameters
        else:
            raise ValueError('num_streams should be %d or %d' %(2,3))


    def train_start(self):
        """switch to train mode
        """
        if self.num_streams==2:
            self.img_enc.train()
            self.aud_enc.train()
            self.aud_rnn.train() ####
            #self.img_rec.train() # do not share the weights of parameters
            #self.aud_rec.train() # do not share the weights of parameters
            #self.feats_rec.train() # share the weights of parameters
        elif self.num_streams==3:
            self.img_enc.train()
            self.txt_enc.train()
            self.aud_enc.train() ####
            self.aud_rnn.train() ####
            #self.img_rec.train() # do not share the weights of parameters
            #self.aud_rec.train() # do not share the weights of parameters
            #self.feats_rec.train() # share the weights of parameters
        else:
            raise ValueError('num_streams should be %d or %d' %(2,3))

    def val_start(self):
        """switch to evaluate mode
        """
        if self.num_streams==2:
            self.img_enc.eval()
            self.aud_enc.eval()
            self.aud_rnn.eval() ####
            #self.img_rec.eval() # do not share the weights of parameters
            #self.aud_rec.eval() # do not share the weights of parameters
            #self.feats_rec.eval() # share the weights of parameters
        elif self.num_streams==3:
            self.img_enc.eval()
            self.txt_enc.eval()
            self.aud_enc.eval() ####
            self.aud_rnn.eval() ####
            #self.img_rec.eval() # do not share the weights of parameters
            #self.aud_rec.eval() # do not share the weights of parameters
            #self.feats_rec.eval() # share the weights of parameters
        else:
            raise ValueError('num_streams should be %d or %d' %(2,3))


    #def forward_emb(self, images, captions, lengths, volatile=False):
    #def forward_emb(self, images, captions, audios, lengths, volatile=False):
    def forward_emb(self, images, audios, captions=None, lengths=None, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if self.num_streams == 3:
            with torch.no_grad():
                images = Variable(images)
                captions = Variable(captions)
                audios = Variable(audios) ####
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                audios = audios.cuda() ####

            # Forward
            img_emb = self.img_enc(images)
            cap_emb = self.txt_enc(captions, lengths)
            aud_emb = self.aud_enc(audios)
            aud_emb = aud_emb.permute(0,2,1)
            lengths = [128 for i in range(aud_emb.shape[0])]
            aud_emb = self.aud_rnn(aud_emb,lengths)
            #aud_emb=l2norm(aud_emb)

            return img_emb, cap_emb, aud_emb
        elif self.num_streams == 2:
            with torch.no_grad():
                images = Variable(images)
                audios = Variable(audios)
            if torch.cuda.is_available():
                images = images.cuda()
                audios = audios.cuda()

            # Forward
            img_emb = self.img_enc(images)
            aud_emb = self.aud_enc(audios)
            aud_emb = aud_emb.permute(0,2,1)
            lengths = [128 for i in range(aud_emb.shape[0])]
            aud_emb = self.aud_rnn(aud_emb,lengths)
            #aud_emb=l2norm(aud_emb)

            return img_emb, aud_emb
        else:
            raise ValueError('num_streams should be %d or %d' %(2,3))


    #def l2norm_loss(self, img_emb, img_emb_1, aud_emb, aud_emb_1, cap_emb=None, cap_emb_1=None, **kwargs):
    def l2norm_loss(self, feats_emb, feats_emb_1):
        """
        compute the euclidean distance of the given pairs
        """

        cost_l2norm=torch.sum(torch.norm(feats_emb_1-feats_emb,2,1)**2)

        return cost_l2norm


    #def forward_loss(self, img_emb, cap_emb, **kwargs):
    #def forward_loss(self, img_emb, cap_emb, aud_emb, **kwargs):
    def forward_loss(self, feats1_emb, feats2_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
           Args:
               -feats1_emb: img_emb, aud_emb or cap_emb
               -feats2_emb: img_emb, aud_emb or cap_emb
           Returns:
               -loss
        """
        loss = self.criterion(feats1_emb, feats2_emb)
        #loss = self.criterion(img_emb, cap_emb)+self.criterion(img_emb, aud_emb)+self.criterion(cap_emb, aud_emb)
        #self.logger.update('Le', loss.item(), img_emb.size(0))
        #self.logger.update('Le', loss.item(), feats1_emb.size(0))
        return loss


    #def train_emb(self, images, captions, lengths, ids=None, *args):
    #def train_emb(self, images, captions, audios, lengths, ids=None, *args):
    def train_emb(self, images, audios, ids=None, captions=None, lengths=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        if self.num_streams==3:
            #img_emb, cap_emb, aud_emb = self.forward_emb(images, captions, audios, lengths)
            img_emb, cap_emb, aud_emb = self.forward_emb(images, audios, captions, lengths)

            img_emb_1rec, aud_emb_1rec, cap_emb_1rec=self.re_feats(img_emb, aud_emb, cap_emb)
            #img_emb_1rec=self.img_rec(img_emb_1rec) # do not share weights of parameters
            #aud_emb_1rec=self.aud_rec(aud_emb_1rec) # do not share weights of parameters
            #img_emb_1rec=self.feats_rec(img_emb_1rec) # share the weights of parameters
            #aud_emb_1rec=self.feats_rec(aud_emb_1rec) # share the weights of parameters
            #cap_emb_1rec=self.feats_rec(cap_emb_1rec) # share the weights of parameters

            img_emb_2rec, aud_emb_2rec, cap_emb_2rec=self.re_feats(img_emb_1rec, aud_emb_1rec, cap_emb_1rec)
            #img_emb_2rec=self.img_rec(img_emb_2rec) # do not share the weights of parameters
            #aud_emb_2rec=self.aud_rec(aud_emb_2rec) # do not share the weights of parameters
            #img_emb_2rec=self.feats_rec(img_emb_2rec) # share the weights of parameters
            #aud_emb_2rec=self.feats_rec(aud_emb_2rec) # share the weights of parameters
            #cap_emb_2rec=self.feats_rec(cap_emb_2rec) # share the weights of parameters

            #self.optimizer.zero_grad()
            cost_base = self.forward_loss(img_emb, aud_emb)+self.forward_loss(img_emb, cap_emb)+self.forward_loss(aud_emb, cap_emb)
            #cost_1rec_l2norm=self.l2norm_loss(img_emb, img_emb_1rec, aud_emb, aud_emb_1rec)
            ##cost_1rec_l2norm=self.l2norm_loss(img_emb, img_emb_1rec, aud_emb, aud_emb_1rec)
            #cost_1rec_l2norm=self.l2norm_loss(img_emb, img_emb_1rec)+self.l2norm_loss(aud_emb, aud_emb_1rec)+self.l2norm_loss(cap_emb, cap_emb_1rec)
            #cost_2rec_l2norm=self.l2norm_loss(img_emb, img_emb_2rec, aud_emb, aud_emb_2rec)
            #cost_2rec_l2norm=self.l2norm_loss(img_emb, img_emb_2rec, aud_emb, aud_emb_2rec, cap_emb, cap_emb_2rec)
            cost_2rec_l2norm=self.l2norm_loss(img_emb, img_emb_2rec)+self.l2norm_loss(aud_emb, aud_emb_2rec)+self.l2norm_loss(cap_emb, cap_emb_2rec)
            #cost_im_1rec_im_s_1rec_s=self.forward_loss(img_emb_1rec, img_emb)+self.forward_loss(aud_emb_1rec, aud_emb)
            #cost_im_1rec_im_s_1rec_s=self.forward_loss(img_emb_1rec, img_emb)+self.forward_loss(aud_emb_1rec, aud_emb)+self.forward_loss(cap_emb_1rec, cap_emb)
            #cost_im_1rec_s_im_s_1rec=self.forward_loss(img_emb_1rec, aud_emb)+self.forward_loss(img_emb, aud_emb_1rec)
            #cost_im_1rec_s_im_s_1rec=self.forward_loss(img_emb_1rec, aud_emb)+self.forward_loss(img_emb_1rec, cap_emb)+self.forward_loss(aud_emb_1rec, img_emb)+self.forward_loss(aud_emb_1rec, cap_emb)+self.forward_loss(cap_emb_1rec, img_emb)+self.forward_loss(cap_emb_1rec, aud_emb)
            self.optimizer.zero_grad()

            lambda1=0.05
            #lambda2=0.05
            # measure accuracy and record loss
            #loss=cost_base
            #loss=cost_base+lambda1*cost_1rec_l2norm
            loss=cost_base+lambda1*cost_2rec_l2norm
            #loss=cost_base+lambda1*(cost_1rec_l2norm+cost_2rec_l2norm)
            #loss=cost_base+lambda1*cost_im_1rec_im_s_1rec_s
            #loss=cost_base+lambda1*cost_im_1rec_s_im_s_1rec
            #loss=cost_base+lambda1*cost_im_1rec_im_s_1rec_s+lambda2*cost_1rec_l2norm
            #loss=cost_base+lambda1*cost_im_1rec_s_im_s_1rec+lambda2*cost_1rec_l2norm
            #loss=cost_base+lambda1*cost_im_1rec_im_s_1rec_s+lambda2*cost_2rec_l2norm
            #loss=cost_base+lambda1*cost_im_1rec_s_im_s_1rec+lambda2*cost_2rec_l2norm
            #loss=cost_base+lambda1*(cost_im_1rec_im_s_1rec_s+cost_im_1rec_s_im_s_1rec)
            #loss=cost_base+lambda1*(cost_im_1rec_im_s_1rec_s+cost_im_1rec_s_im_s_1rec)+cost_1rec_l2norm
            #loss=cost_base+lambda1*(cost_im_1rec_im_s_1rec_s+cost_im_1rec_s_im_s_1rec)+cost_2rec_l2norm
            #loss=cost_base+lambda1*(cost_im_1rec_im_s_1rec_s+cost_im_1rec_s_im_s_1rec)+lambda2*(cost_1rec_l2norm+cost_2rec_l2norm)
            self.logger.update('Le', loss.item(), img_emb.size(0))
        elif self.num_streams==2:
            img_emb, aud_emb = self.forward_emb(images,audios)
            img_emb_1rec, aud_emb_1rec=self.re_feats(img_emb, aud_emb)
            #img_emb_1rec=self.img_rec(img_emb_1rec) # do not share the weights of parameters
            #aud_emb_1rec=self.aud_rec(aud_emb_1rec) # do not share the weights of parameters
            #img_emb_1rec=self.feats_rec(img_emb_1rec) # share the weights of parameters
            #aud_emb_1rec=self.feats_rec(aud_emb_1rec) # share the weights of parameters

            img_emb_2rec, aud_emb_2rec=self.re_feats(img_emb_1rec, aud_emb_1rec)
            #img_emb_2rec=self.img_rec(img_emb_2rec) # do not share the weights of parameters
            #aud_emb_2rec=self.aud_rec(aud_emb_2rec) # do not share the weights of parameters
            #img_emb_2rec=self.feats_rec(img_emb_2rec) # share the weights of parameters
            #aud_emb_2rec=self.feats_rec(aud_emb_2rec) # share the weights of parameters

            #self.optimizer.zero_grad()
            cost_base = self.forward_loss(img_emb, aud_emb)
            #cost_1rec_l2norm=self.l2norm_loss(img_emb, img_emb_1rec)+self.l2norm_loss(aud_emb, aud_emb_1rec)
            cost_2rec_l2norm=self.l2norm_loss(img_emb, img_emb_2rec)+self.l2norm_loss(aud_emb, aud_emb_2rec)
            #cost_im_1rec_im_s_1rec_s=self.forward_loss(img_emb_1rec, img_emb)+self.forward_loss(aud_emb_1rec, aud_emb)
            #cost_im_1rec_s_im_s_1rec=self.forward_loss(img_emb_1rec, aud_emb)+self.forward_loss(img_emb, aud_emb_1rec)
            self.optimizer.zero_grad()
            lambda1=0.05
            #lambda2=0.05

            # measure accuracy and record loss
            #loss=cost_base
            #loss=cost_base+lambda1*cost_1rec_l2norm
            loss=cost_base+lambda1*cost_2rec_l2norm
            #loss=cost_base+lambda1*(cost_1rec_l2norm+cost_2rec_l2norm)
            #loss=cost_base+lambda1*cost_im_1rec_im_s_1rec_s
            #loss=cost_base+lambda1*cost_im_1rec_s_im_s_1rec
            #loss=cost_base+lambda1*cost_im_1rec_im_s_1rec_s+lambda2*cost_1rec_l2norm
            #loss=cost_base+lambda1*cost_im_1rec_s_im_s_1rec+lambda2*cost_1rec_l2norm
            #loss=cost_base+lambda1*cost_im_1rec_im_s_1rec_s+lambda2*cost_2rec_l2norm
            #loss=cost_base+lambda1*cost_im_1rec_s_im_s_1rec+lambda2*cost_2rec_l2norm
            #loss=cost_base+lambda1*(cost_im_1rec_im_s_1rec_s+cost_im_1rec_s_im_s_1rec)
            #loss=cost_base+lambda1*(cost_im_1rec_im_s_1rec_s+cost_im_1rec_s_im_s_1rec)+cost_1rec_l2norm
            #loss=cost_base+lambda1*(cost_im_1rec_im_s_1rec_s+cost_im_1rec_s_im_s_1rec)+cost_2rec_l2norm
            #loss=cost_base+lambda1*(cost_im_1rec_im_s_1rec_s+cost_im_1rec_s_im_s_1rec)+lambda2*(cost_1rec_l2norm+cost_2rec_l2norm)
            self.logger.update('Le', loss.item(), img_emb.size(0))


        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            #clip_grad_norm(self.params, self.grad_clip)
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
