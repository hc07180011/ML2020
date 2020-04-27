import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda")

class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        self.root = root
        self.label = torch.from_numpy(label).long()
        self.transforms = transforms
        self.fnames = []

        for i in range(200):
            self.fnames.append('{:03d}'.format(i))

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        img = self.transforms(img)
        label = self.label[idx]
        return img, label

    def __len__(self):
        return 200

class Attacker:
    def __init__(self, model, img_dir, label):
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        self.dataset = Adverdataset(img_dir, label, transform)

        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    def fgsm_attack(self, image, epsilon, target, T):
        for i in range(T):
            self.model.zero_grad()
            output = self.model(image)
            if output.max(1, keepdim=True)[1].item() != target.item():
                break
            loss = F.cross_entropy(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = image.grad.detach()
            sign_data_grad = data_grad.sign()
            perturbed_image = image + epsilon * sign_data_grad
            image = perturbed_image.detach()
            image.requires_grad = True
        return image, i

    def attack(self, epsilon):

        adv_examples = []
        data_raws = []
        result = []
        fail, success = 0, 0

        for idx, (data, target) in enumerate(self.loader):

            data, target = data.to(device), target.to(device)
            data_raw = data
            data.requires_grad = True
            data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            data_raw = data_raw.squeeze().detach().cpu().numpy()
            data_raws.append(data_raw)

            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]
            if init_pred.item() != target.item():
                success += 1
                adv_examples.append(data_raw)
                continue

            if sys.argv[3] == 'fgsm':
                perturbed_data, T = self.fgsm_attack(data, 0.01, target, 1)
            elif sys.argv[3] == 'best':
                perturbed_data, T = self.fgsm_attack(data, epsilon, target, 100)

            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]

            adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            adv_ex = adv_ex.squeeze().detach().cpu().numpy()

            if final_pred.item() == target.item():
                fail += 1
                adv_examples.append(data_raw)
            else:
                success += 1
                adv_examples.append(adv_ex)

        final_acc = (fail / (success + fail))

        print(final_acc)
        return np.clip((np.transpose(adv_examples, (0, 2, 3, 1)) * 255.0).astype(np.uint8), 0, 255), \
                np.clip((np.transpose(data_raws, (0, 2, 3, 1)) * 255.0).astype(np.uint8), 0, 255)

if __name__ == '__main__':

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    assert sys.argv[3] == 'fgsm' or sys.argv[3] == 'best'
    df = pd.read_csv(os.path.join(input_path, 'labels.csv'))
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(os.path.join(input_path, 'categories.csv'))
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    epsilon = 0.001

    attacker = Attacker(models.densenet121(pretrained = True), os.path.join(input_path, 'images'), df)

    ex, raw = attacker.attack(epsilon)

    for i, img in enumerate(ex):
        im = Image.fromarray(img)
        im.save(os.path.join(output_path, '{:03d}'.format(i) + '.png'))

    print(np.mean(np.max(np.abs(np.array(ex.astype(np.int32) - raw.astype(np.int32))), axis=(-1, -2, -3))))
