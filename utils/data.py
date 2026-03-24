import PIL.Image as PImage
from torchvision.transforms import InterpolationMode, transforms
from utils.restoration_loader import BaseDataset
from utils.paired_transform import PairedTransform


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

# build_dataset has been modified a lot
def build_dataset(
        data_path_train: str, data_path_val: str, final_reso: int,
        hflip=False, mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize((final_reso, final_reso), interpolation=InterpolationMode.LANCZOS),

        transforms.ToTensor(), normalize_01_into_pm1,
    ]

    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())

    # Create paired transformations
    train_aug = PairedTransform(train_aug)
    val_aug = PairedTransform(val_aug)
    # build dataset
    train_set = BaseDataset(json_path=data_path_train, transform=train_aug)
    val_set = BaseDataset(json_path=data_path_val, transform=val_aug)
    num_classes = 1000 #  fix num_classes
    print(f'[Dataset] {data_path_val}, Train (ignore if testing): {len(train_set)}, Val/Test: {len(val_set)}')

    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
