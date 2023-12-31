# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import json
import librosa
import nltk
import numpy as np
import os
from PIL import Image
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class ImageCaptionDataset(Dataset):
    #def __init__(self, dataset_json_file, audio_conf=None, image_conf=None):
    def __init__(self, dataset_json_file, num_streams, vocab, audio_conf=None, image_conf=None):
        """
        Dataset that manages a set of paired images and audio recordings

        :param dataset_json_file
        :param audio_conf: Dictionary containing the sample rate, window and
        the window length/stride in seconds, and normalization to perform (optional)
        :param image_transform: torchvision transform to apply to the images (optional)
        """
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        #self.image_base_path = data_json['image_base_path']
        self.image_base_path = "/root/wenlong_workspace/code/VASE/data/data/vision/torralba/deeplearning/images256/" # change to places image path
        #self.audio_base_path = data_json['audio_base_path']
        self.audio_base_path = "/root/wenlong_workspace/code/VASE/data/PlacesAudio_400k_distro/" # change to places audio path

        self.num_streams=num_streams
        self.vocab=vocab

        if not audio_conf:
            self.audio_conf = {}
        else:
            self.audio_conf = audio_conf

        if not image_conf:
            self.image_conf = {}
        else:
            self.image_conf = image_conf

        crop_size = self.image_conf.get('crop_size', 224)
        center_crop = self.image_conf.get('center_crop', False)

        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.RandomResizedCrop(crop_size), transforms.ToTensor()])

        RGB_mean = self.image_conf.get('RGB_mean', [0.485, 0.456, 0.406])
        RGB_std = self.image_conf.get('RGB_std', [0.229, 0.224, 0.225])
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

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

    def _LoadImage(self, impath):
        img = Image.open(impath).convert('RGB')
        img = self.image_resize_and_crop(img)
        img = self.image_normalize(img)
        return img

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data[index]
        wavpath = os.path.join(self.audio_base_path, datum['wav'])
        imgpath = os.path.join(self.image_base_path, datum['image'])
        audio, nframes = self._LoadAudio(wavpath)
        image = self._LoadImage(imgpath)

        if self.num_streams==2:
            #return image, audio, index, nframes
            #return image, audio, nframes, index
            return image, audio, index, nframes
        elif self.num_streams==3:
            caption=datum['asr_text']
            tokens=nltk.tokenize.word_tokenize(str(caption.encode('utf-8')).lower().decode('utf-8'))
            caption=[]
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            target=torch.Tensor(caption)
            #return image, target, audio, index, nframes
            #return image, target, audio, nframes, index
            return image, target, audio, index, nframes
        else:
            raise ValueError('num_streams should be %d or %d' %(2,3))

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
        - image: torch tensor of shape (3, 256, 256).
        - caption: torch tensor of shape(?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    #images, captions, audios, ids, nframes=zip(*data)
    images, captions, audios, nframes, ids=zip(*data)

    # Merge images and audios (convert tuple of 3D tensor to 4D tensor)
    images=torch.stack(images, 0)
    audios=torch.stack(audios, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths=[len(cap) for cap in captions]
    targets=torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end=lengths[i]
        targets[i, :end]=cap[:end]

    return images, targets, audios, lengths, nframes, ids


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
    # Sort a data list by audio length
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, audios, ids, nframes = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    audios = torch.stack(audios, 0)

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
    images, captions, audios, ids, nframes = zip(*data)

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

