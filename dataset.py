from paddle.io import Dataset
import os
from paddle.vision import image_load


class ImageNet2012Dataset(Dataset):

    def __init__(self, file_folder, mode="train", transform=None):
        """Init ImageNet2012 Dataset with dataset file path, mode(train/val), and transform"""
        super(ImageNet2012Dataset, self).__init__()
        assert mode in ["train", "val"]
        self.file_folder = file_folder
        self.transform = transform
        self.img_path_list = []
        self.label_list = []

        if mode == "train":
            self.list_file = "ILSVRC2012_w/train.txt"
        else:
            self.list_file = "ILSVRC2012_w/val_list.txt"

        with open(self.list_file, 'r') as infile:
            for line in infile:
                img_path = line.strip().split()[0]
                img_label = int(line.strip().split()[1])
                self.img_path_list.append(os.path.join(self.file_folder, img_path))
                self.label_list.append(img_label)
        print(f'----- Imagenet2012 image {mode} list len = {len(self.label_list)}')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        data = image_load(self.img_path_list[index]).convert('RGB')
        data = self.transform(data)
        label = self.label_list[index]

        return data, label