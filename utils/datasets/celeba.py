import numpy as np
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import pickle

tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
])

class CelebA(Dataset):
    def __init__(self, dataframe, folder_dir, target_id, transform=None, gender=None, target=None, ret_idxs=False):
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.target_id = target_id
        self.transform = transform
        self.ret_idxs = ret_idxs
        self.file_names = dataframe.index
        self.labels = np.concatenate(dataframe.labels.values).astype(float)
        gender_id = 20
        self.gender_id = gender_id

        if gender is not None:
            if target is not None:
                label_np = np.concatenate(dataframe.labels.values)
                gender_idx = np.where(label_np[:, gender_id] == gender)[0]
                target_idx = np.where(label_np[:, target_id] == target)[0]
                idx = list(set(gender_idx) & set(target_idx))
                self.file_names = self.file_names[idx]
                self.labels = np.concatenate(dataframe.labels.values[idx]).astype(float)
            else:
                label_np = np.concatenate(dataframe.labels.values)
                gender_idx = np.where(label_np[:, gender_id] == gender)[0]
                self.file_names = self.file_names[gender_idx]
                self.labels = np.concatenate(dataframe.labels.values[gender_idx]).astype(float)

    def __len__(self):
        return len(self.labels)

    def get_multiple_items(self, indices):
        images = []
        target_labels = []
        gender_labels = []

        for index in indices:
            image = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
            image = self.transform(image)
            label = self.labels[index]
            images.append(image)
            target_labels.append(label[self.target_id])
            gender_labels.append(label[self.gender_id])

        if self.ret_idxs:
            return indices, images, target_labels, gender_labels

        return images, target_labels, gender_labels
    def __getitem__(self, index):
        if type(index) == list or type(index) == np.ndarray:
            return self.get_multiple_items(index)

        with open(os.path.join(self.folder_dir, "transformed", self.file_names[index]), 'rb') as fh:
            image = pickle.load(fh)

        label = self.labels[index]

        if self.ret_idxs:
            return index, image, label[self.target_id], label[self.gender_id]
        return image, label[self.target_id], label[self.gender_id]


def get_dataset(df, data_path, target_id):
    return CelebA(df, data_path, target_id, transform=tfms)

def get_loader(df, data_path, target_id, batch_size, gender=None, target=None, ret_idxs=False, num_workers=0):
    dl = CelebA(df, data_path, target_id, transform=tfms, gender=gender, target=target, ret_idxs=ret_idxs)

    pre_fact = 4 if num_workers > 0 else None
    if 'train' in data_path:
        dloader = torch.utils.data.DataLoader(dl, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True, prefetch_factor=pre_fact)
    else:
        dloader = torch.utils.data.DataLoader(dl, shuffle=False, batch_size=batch_size, num_workers=num_workers, prefetch_factor=pre_fact)

    return dloader
