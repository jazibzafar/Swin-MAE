import albumentations as A
import os
from tifffile import imread
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import get_worker_info
from itertools import islice
import torch
import random
from osgeo import gdal
import webdataset as wds


class UsualTransform:
    def __init__(self, input_size):
        self.input_size = input_size

        self.transforms = A.Compose([
            A.RandomCrop(height=self.input_size,
                         width=self.input_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=(0.2, 0.3),
                                       contrast_limit=(0.2, 0.3),
                                       p=0.2),
            A.RandomGamma(gamma_limit=(100, 140), p=0.2),
            A.RandomToneCurve(scale=0.1, p=0.2)
        ])

    def __call__(self, image):
        # Make sure image is a np array
        if type(image) == 'torch.Tensor':
            image = image.numpy()

        crop = ToTensor()(self.transforms(image=image)['image'])
        return crop


class GeoWebDataset(IterableDataset):
    def __init__(self,
                 *,
                 root,
                 n_bands,
                 augmentations,
                 num_nodes=1,
                 num_shards=100,
                 imgs_per_shard=250):
        self.root = root
        self.n_bands = n_bands
        self.augmentations = augmentations
        self.num_nodes = num_nodes
        self.num_shards = num_shards
        self.imgs_per_shard = imgs_per_shard
        self.cropsize = 224
        #
        self.num_patches = 1000000000000  # set it to sth really high for now, so that the generator doesnt get exhausted during trainng

        self.dataset = wds.DataPipeline(wds.ResampledShards(self.root),
                                        wds.split_by_node,
                                        wds.split_by_worker,
                                        self.split_by_dataloader_worker,
                                        # self.printer,
                                        wds.shuffle(8),
                                        wds.tarfile_to_samples(),
                                        wds.to_tuple("tif"),
                                        wds.map(GeoWebDataset.preprocess),
                                        self.slicer,
                                        wds.shuffle(100),  # buffer of size 100
                                        wds.map(self.augmentations),
                                        ).with_length(self.num_patches)

    @staticmethod
    def read_geotif_from_bytestream(data: bytes) -> np.ndarray:
        gdal.FileFromMemBuffer("/vsimem/tmp", data)
        ds = gdal.Open("/vsimem/tmp")
        bands = ds.RasterCount
        ys = ds.RasterYSize
        xs = ds.RasterXSize
        # arr = np.empty((bands, ys, xs), dtype="float32")  # CHW
        # for b in range(1, bands + 1):
        #     band = ds.GetRasterBand(b)
        #     arr[b - 1, :, :] = band.ReadAsArray()
        # return torch.from_numpy(arr) / 255
        arr = np.empty((ys, xs, bands), dtype="uint8")  # HWC
        for b in range(1, bands + 1):
            band = ds.GetRasterBand(b)
            arr[:, :, b - 1] = band.ReadAsArray()
        return arr

    @staticmethod
    def preprocess(sample):
        return GeoWebDataset.read_geotif_from_bytestream(sample[0])

    @staticmethod
    def slice_image(samples, tilesize: int):
        for img in samples:
            for y in range(0, img.shape[1], tilesize):
                for x in range(0, img.shape[2], tilesize):
                    yield img[:, y:y + tilesize, x:x + tilesize]  # CHW

    @staticmethod
    def split_by_dataloader_worker(iterable):
        worker_info = get_worker_info()
        # print("Worker info: ", worker_info)
        if worker_info is None:
            return iterable
        else:
            worker_num = worker_info.num_workers
            worker_id = worker_info.id
            sliced_data = islice(iterable, worker_id, None, worker_num)
            return sliced_data

    @staticmethod
    def printer(iterable):
        for x in iterable:
            print("From node X, worker Y, dataloader worker Z: ")
            print(x)
            yield x

    def slicer(self, img):
        return GeoWebDataset.slice_image(img, self.cropsize)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.imgs_per_shard * self.num_shards * 100  # each image has 100 crops.