import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import librosa
import pdb
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
import scipy.signal

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])
                                

def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f8k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        # if ids provided by get_paths, use split-specific ids
        if ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root, caption, img_id, path, image = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def get_raw_item(self, index):
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')

        return root, caption, img_id, path, image

    def __len__(self):
        return len(self.ids)


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)

class Flickr8kDataset(data.Dataset):
    """
    Dataset loader for Flickr8k full datasets.
    """

    #def __init__(self, root, json, split, vocab, transform=None):
    def __init__(self, root, json, split, vocab, num_streams, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.num_streams = num_streams
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []

        self.audio_path='/root/wenlong_workspace/code/VASE/flickr_audio/' #### change to your flickr audio path

        self.audio_conf = {}

        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

        #pdb.set_trace()

        self.windows = {'hamming': scipy.signal.hamming,
        'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
            'bartlett': scipy.signal.bartlett}


    def _LoadAudio(self, path):
        audio_type = self.audio_conf.get('audio_type', 'melspectrogram')
        if audio_type not in ['melspectrogram', 'spectrogram']:
            raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        preemph_coef = self.audio_conf.get('preemph_coef', 0.97)
        sample_rate = self.audio_conf.get('sample_rate', 16000)
        window_size = self.audio_conf.get('window_size', 0.025)
        window_stride = self.audio_conf.get('window_stride', 0.01)
        window_type = self.audio_conf.get('window_type', 'hamming')
        num_mel_bins = self.audio_conf.get('num_mel_bins', 40)
        target_length = self.audio_conf.get('target_length', 2048)
        use_raw_length = self.audio_conf.get('use_raw_length', False)
        padval = self.audio_conf.get('padval', 0)
        fmin = self.audio_conf.get('fmin', 20)
        n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        y, sr = librosa.load(path, sample_rate)
        if y.size == 0:
            y = np.zeros(200)
        y = y - y.mean()
        y = preemphasis(y, preemph_coef)
        # compute mel spectrogram
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length,
            window=self.windows.get(window_type, self.windows['hamming']))
        spec = np.abs(stft)**2
        if audio_type == 'melspectrogram':
            mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
            melspec = np.dot(mel_basis, spec)
            logspec = librosa.power_to_db(melspec, ref=np.max)
        elif audio_type == 'spectrogram':
            logspec = librosa.power_to_db(spec, ref=np.max)
        n_frames = logspec.shape[1]
        if use_raw_length:
            target_length = n_frames
        p = target_length - n_frames
        if p > 0:
            logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
                constant_values=(padval,padval))
        elif p < 0:
            logspec = logspec[:,0:p]
            n_frames = target_length
        logspec = torch.FloatTensor(logspec)
        return logspec, n_frames

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        #wave_path=os.path.join(self.audio_path,'waves',path.split('.'[0],'_',ann_id[1],'.wav'))
        wave_path=self.audio_path + 'wavs/' + path.split('.')[0] + '_' + str(ann_id[1]) + '.wav'
        #pdb.set_trace()
        if self.transform is not None:
            image = self.transform(image)
        audio, nframes=self._LoadAudio(wave_path)
        #audio=torch.FloatTensor(audio)
        # Convert caption (string) to word ids.
        if self.num_streams==3:
            tokens = nltk.tokenize.word_tokenize(
                str(caption).lower().decode('utf-8'))
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            #pdb.set_trace()
            #return image, target, index, img_id, audio
            return image, target, audio, index, img_id
        elif self.num_streams==2:
            return image, audio, index, img_id
        else:
            raise ValueError('num_streams should be %d or %d' %(2,3))

    def __len__(self):
        return len(self.ids)

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length


#def collate_fn(data):
#    """Build mini-batch tensors from a list of (image, caption) tuples.
#    Args:
#        data: list of (image, caption) tuple.
#            - image: torch tensor of shape (3, 256, 256).
#            - caption: torch tensor of shape (?); variable length.
#
#    Returns:
#        images: torch tensor of shape (batch_size, 3, 256, 256).
#        targets: torch tensor of shape (batch_size, padded_length).
#        lengths: list; valid length for each padded caption.
#    """
#    # Sort a data list by caption length
#    data.sort(key=lambda x: len(x[1]), reverse=True)
#    #images, captions, ids, img_ids = zip(*data)
#    #images, captions, ids, img_ids, audios = zip(*data)
#    images, captions, audios, ids, img_ids = zip(*data)
#
#    # Merge images (convert tuple of 3D tensor to 4D tensor)
#    images = torch.stack(images, 0)
#    audios = torch.stack(audios, 0)
#
#    # Merget captions (convert tuple of 1D tensor to 2D tensor)
#    lengths = [len(cap) for cap in captions]
#    targets = torch.zeros(len(captions), max(lengths)).long()
#    for i, cap in enumerate(captions):
#        end = lengths[i]
#        targets[i, :end] = cap[:end]
#
#    #return images, targets, lengths, ids, audios
#    return images, targets, audios, lengths, ids


def collate_fn_two_streams(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    #images, captions, ids, img_ids = zip(*data)
    #images, captions, ids, img_ids, audios = zip(*data)
    images, audios, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    audios = torch.stack(audios, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    #lengths = [len(cap) for cap in captions]
    #targets = torch.zeros(len(captions), max(lengths)).long()
    #for i, cap in enumerate(captions):
    #    end = lengths[i]
    #    targets[i, :end] = cap[:end]

    #return images, targets, lengths, ids, audios
    return images, audios, ids

def collate_fn_three_streams(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    #images, captions, ids, img_ids = zip(*data)
    #images, captions, ids, img_ids, audios = zip(*data)
    images, captions, audios, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    audios = torch.stack(audios, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    #return images, targets, lengths, ids, audios
    #return images, targets, audios, lengths, ids
    return images, audios, ids, targets, lengths

#def get_loader_single(data_name, split, root, json, vocab, transform,
#                      batch_size=100, shuffle=True,
#                      num_workers=2, ids=None, collate_fn=collate_fn):
def get_loader_single(opt, split, root, json, vocab, num_streams, transform,
                      batch_size=100, shuffle=True,
                      num_workers=1, ids=None, collate_fn=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if num_streams==2:
        collate_fn=collate_fn_two_streams
    elif num_streams==3:
        collate_fn=collate_fn_three_streams  
    else:
        raise ValueError('num_streams should be %d or %d' %(2,3))

    if 'coco' in opt.data_name:
        # COCO custom dataset
        dataset = CocoDataset(root=root,
                              json=json,
                              vocab=vocab,
                              transform=transform, ids=ids)
    elif 'f8k' in opt.data_name or 'f30k' in opt.data_name:
        #dataset = FlickrDataset(root=root,
        #                        split=split,
        #                        json=json,
        #                        vocab=vocab,
        #                        transform=transform)
        #dataset = Flickr8kDataset(root=root,
        #                        split=split,
        #                        json=json,
        #                        vocab=vocab,
        #                        transform=transform)
        dataset = Flickr8kDataset(root=root,
                                split=split,
                                json=json,
                                vocab=vocab,
                                num_streams=num_streams,
                                transform=transform)

    # Data loader
    #data_loader = torch.utils.data.DataLoader(dataset=dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=shuffle,
    #                                          pin_memory=True,
    #                                          num_workers=num_workers,
    #                                          collate_fn=collate_fn)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              worker_init_fn=opt.worker_init_fn)

    return data_loader


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=1):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    #data_loader = torch.utils.data.DataLoader(dataset=dset,
    #                                          batch_size=batch_size,
    #                                          shuffle=shuffle,
    #                                          pin_memory=True,
    #                                          collate_fn=collate_fn)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              worker_init_fn=opt.worker_init_fn)
    return data_loader


def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                          batch_size, True, workers)
        val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                        batch_size, False, workers)
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, 'train', opt)
        #train_loader = get_loader_single(opt.data_name, 'train',
        train_loader = get_loader_single(opt, 'train',
                                         roots['train']['img'],
                                         roots['train']['cap'],
                                         vocab, transform, ids=ids['train'],
                                         batch_size=batch_size, shuffle=True,
                                         num_workers=workers,
                                         collate_fn=None)

        transform = get_transform(data_name, 'val', opt)
        #val_loader = get_loader_single(opt.data_name, 'val',
        val_loader = get_loader_single(opt, 'val',
                                       roots['val']['img'],
                                       roots['val']['cap'],
                                       vocab, transform, ids=ids['val'],
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=None)

    return train_loader, val_loader


def get_loaders_two_three_streams(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                          batch_size, True, workers)
        val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                        batch_size, False, workers)
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, 'train', opt)
        #train_loader = get_loader_single(opt.data_name, 'train',
        train_loader = get_loader_single(opt, 'train',
                                         roots['train']['img'],
                                         roots['train']['cap'],
                                         vocab, opt.num_streams, transform, ids=ids['train'],
                                         batch_size=batch_size, shuffle=True,
                                         num_workers=workers,
                                         collate_fn=None)

        transform = get_transform(data_name, 'val', opt)
        #val_loader = get_loader_single(opt.data_name, 'val',
        val_loader = get_loader_single(opt, 'val',
                                       roots['val']['img'],
                                       roots['val']['cap'],
                                       vocab, opt.num_streams, transform, ids=ids['val'],
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=None)

    return train_loader, val_loader

def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)

    if opt.num_streams==2:
        collate_fn=collate_fn_two_streams
    elif opt.num_streams==3:
        collate_fn=collate_fn_three_streams  
    else:
        raise ValueError('num_streams should be %d or %d' %(2,3))

    if opt.data_name.endswith('_precomp'):
        test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                         batch_size, False, workers)
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, split_name, opt)
        #test_loader = get_loader_single(opt.data_name, split_name,
        test_loader = get_loader_single(opt, split_name,
                                        roots[split_name]['img'],
                                        roots[split_name]['cap'],
                                        vocab, opt.num_streams, transform, ids=ids[split_name],
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=workers,
                                        collate_fn=collate_fn)

    return test_loader
