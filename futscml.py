import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image as pilmage
import numpy as np

from torchvision import transforms

class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# assume -1 to 1 tensor
def tensor_to_pil(tensor):
    return pilmage.fromarray(((tensor.clamp(-1, 1) + 1.) * 127.5).detach().squeeze().\
        permute(1,2,0).data.cpu().numpy().astype(np.uint8))

def pil_to_tensor(pil, transform=None, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    tensor = transform(pil).unsqueeze(0)
    return tensor

def np_to_pil(npy):
    return pilmage.fromarray(npy.astype(np.uint8))

def pil_to_np(pil):
    return np.array(pil)

def tensor_to_np(tensor, cut_dim_to_3=True):
    if len(tensor.shape) == 4:
        if cut_dim_to_3:
            tensor = tensor[0]
        else:
            return tensor.data.cpu().numpy().transpose((0, 2, 3, 1))
    return tensor.data.cpu().numpy().transpose((1,2,0))

def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)

def pil_resize_short_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_short = (trg_size / pil.width) if short_w else (trg_size / pil.height)
    resized = pil.resize((int(pil.width * ar_resized_short), int(pil.height * ar_resized_short)), pilmage.BICUBIC)
    return resized

def pil_resize_long_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_long = (trg_size / pil.height) if short_w else (trg_size / pil.width)
    resized = pil.resize((int(pil.width * ar_resized_long), int(pil.height * ar_resized_long)), pilmage.BICUBIC)
    return resized
    
class ImageTensorConverter:
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], resize=None, is_bgr=False, 
                    mul_by=None, unsqueeze=False, device=None, clamp_to_pil=None, drop_alpha=False):
        self.mean = mean
        self.std = std
        self.resize = resize
        self.to_tensor_transform = []
        self.inverse_transform = []
        if drop_alpha:
            self.to_tensor_transform.append(transforms.Lambda(lambda x: x.convert('RGB')))
        if resize is not None:
            self.to_tensor_transform.append(transforms.Resize(resize))
        self.to_tensor_transform.append(transforms.ToTensor())
        self.inverse_transform.append(transforms.ToPILImage())
        if clamp_to_pil is not None:
            self.inverse_transform.append(transforms.Lambda(lambda x: x.clamp(min=clamp_to_pil[0], max=clamp_to_pil[1])))
        if is_bgr:
            self.to_tensor_transform.append(transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]))
            self.inverse_transform.append(transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]))
            self.mean = [self.mean[2], self.mean[1], self.mean[0]]
            self.std = [self.std[2], self.std[1], self.std[0]]
        self.to_tensor_transform.append(transforms.Normalize(self.mean, self.std))
        self.inverse_transform.append(transforms.Lambda(lambda x: x + torch.tensor(self.mean).view(-1, 1, 1)))
        self.inverse_transform.append(transforms.Lambda(lambda x: x * torch.tensor(self.std).view(-1, 1, 1)))
        if mul_by:
            self.to_tensor_transform.append(transforms.Lambda(lambda x: x.mul_(mul_by)))
            self.inverse_transform.append(transforms.Lambda(lambda x: x.div_(mul_by)))
        if unsqueeze:
            self.to_tensor_transform.append(transforms.Lambda(lambda x: x.unsqueeze(0)))
            # not sure if I should assume squeeze in inverse
            self.inverse_transform.append(transforms.Lambda(lambda x: x.squeeze()))
        if device:
            self.to_tensor_transform.append(transforms.Lambda(lambda x: x.to(device)))
            self.inverse_transform.append(transforms.Lambda(lambda x: x.to('cpu')))
        self.to_tensor_transform = transforms.Compose(self.to_tensor_transform)
        self.inverse_transform.reverse() # Reverse doesn't return itself /facepalm
        self.to_pil_transform = transforms.Compose(self.inverse_transform)

    def __call__(self, input):
        if type(input).__name__ == 'Tensor':
            return self.get_pil(input)
        else:
            return self.get_tensor(input)

    def get_tensor(self, input):
        return self.to_tensor_transform(input)

    def get_pil(self, input):
        return self.to_pil_transform(input)

def tensor_to_image(tensor, fname, ext='png'):
    tensor_to_pil(tensor).save(fname if fname.endswith(ext) else f'{fname}.{ext}')
    return fname if fname.endswith(ext) else f'{fname}.{ext}'

def pil_loader(path):
    with open(path, 'rb') as f:
        img = pilmage.open(f)
        return img.convert('RGB')

def is_image(fname):
    fname = fname.lower()
    exts = ['jpg', 'png', 'bmp', 'jpeg', 'tiff']
    ok = any([fname.endswith(ext) for ext in exts])
    return ok

def images_in_directory(dir):
    ls = os.listdir(dir)
    return sorted(list(filter(is_image, ls)))

def subdirectories(dir, ignore_dirs_starting_with_dot=True):
    ls = os.listdir(dir)
    filt_fn = lambda x: os.path.isdir(os.path.join(dir, x))
    if ignore_dirs_starting_with_dot:
        prev = filt_fn
        filt_fn = lambda x: prev(x) and not x.startswith('.')
    return sorted(
        list(filter(
                lambda x: filt_fn(x), 
                ls)
            )
    )
    
class ImageDirectory(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self._root = root
        self.root_path = os.path.expanduser(root)
        self.transform = transform
        self.files = sorted(os.listdir(self.root_path))
        def is_valid_file(path):
            return is_image(path) and not path.lower().startswith('.')
        self.files = list(filter(is_valid_file, self.files))
        self.loader = pil_loader
        print(f"{self.__class__.__name__}: Found {len(self)} images in {root}")
        
    def __getitem__(self, index):
        path = os.path.join(self.root_path, self.files[index])
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.files)

class GramMatrix(nn.Module):
    def forward(self, x):
        n, c, h, w = x.shape
        M = x.view(n, c, h*w)
        G = torch.bmm(M, M.transpose(1,2)) 
        G.div_(h*w*c)
        return G

class GramMatrixMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        self.gm = GramMatrix()

    def forward(self, x, y):
        out = self.loss(self.gm(x), y)
        return out

class ChannelwiseGaussianBlur(nn.Module):
    def __init__(self):
        super().__init__()
        self.cached_filter = None
        self.d2gaussian = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float) / 16.
        self.d2gaussian = self.d2gaussian.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        if self.cached_filter is None or x.shape[1] != self.cached_filter.shape[0]:
            self.cached_filter = self.d2gaussian.repeat(x.shape[1], 1, 1, 1).to(x.device)
        return F.conv2d(x, self.cached_filter, groups=x.shape[1], padding=1)

class ChannelwiseSobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cached_filter1 = None
        self.cached_filter2 = None
        self.xsobel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float)
        self.ysobel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float)
        self.xsobel = self.xsobel.unsqueeze(0).unsqueeze(0)
        self.ysobel = self.ysobel.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        if self.cached_filter1 is None or x.shape[1] != self.cached_filter1.shape[0]:
            self.cached_filter1 = self.xsobel.repeat(x.shape[1], 1, 1, 1).to(x.device)
        if self.cached_filter2 is None or x.shape[1] != self.cached_filter2.shape[0]:
            self.cached_filter2 = self.ysobel.repeat(x.shape[1], 1, 1, 1).to(x.device)
        return F.conv2d(x, self.cached_filter1, groups=x.shape[1], padding=1), \
                F.conv2d(x, self.cached_filter2, groups=x.shape[1], padding=1)

class ChannelwiseSobelMagnitude(nn.Module):
    def __init__(self):
        super().__init__()
        self.dir_sobel = ChannelwiseSobel()

    def forward(self, i):
        x, y = self.dir_sobel(i)
        return (x**2 + y**2) ** 0.5

class ChannelwiseLaplace(nn.Module):
    def __init__(self):
        super().__init__()
        self.cached_filter = None
        self.laplacian = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float)
        self.laplacian = self.laplacian.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        if self.cached_filter is None or x.shape[1] != self.cached_filter.shape[0]:
            self.cached_filter = self.laplacian.repeat(x.shape[1], 1, 1, 1).to(x.device)
        return F.conv2d(x, self.cached_filter, groups=x.shape[1], padding=1)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def calculate_accuracy_and_loss(net, dataloader, device, loss_criterion=nn.CrossEntropyLoss,
                                    output_to_pred=lambda x: torch.argmax(x, 1)):
    top1_total, top5_total, total, running_loss = 0., 0., 0., 0.

    net.to(device).eval()
    criterion = loss_criterion()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            top1, top5 = accuracy(outputs, labels, topk=(1,5))
            top1_total += top1 / (100. / labels.size(0))
            top5_total += top5 / (100. / labels.size(0))

            total += labels.size(0)
            #predicted = output_to_pred(outputs.data)
            #correct_prediction = (predicted == labels)
            #correct += correct_prediction.sum().item()

    return (top1_total.item() / total) * 100, (top5_total.item() / total)*100, running_loss / len(dataloader)

def notebook_progress_bar(progress):
    from IPython.display import clear_output
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def notebook_interactive_text(format, *phs):
    from IPython.display import clear_output

    clear_output(wait = True)
    text = format % tuple(phs)
    print(text)
    
def torch_clear_gpu_mem():
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    