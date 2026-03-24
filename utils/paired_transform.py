import random
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

class PairedTransform:
    def __init__(self, base_transforms):
        """
        Initialize with a list of transformations.
        The transformations should support being applied to both images.
        """
        self.base_transforms = base_transforms

    def __call__(self, img1, img2):
        """
        Apply the transformations to both images.
        """
        for transform in self.base_transforms:
            if isinstance(transform, transforms.Resize):
                img1 = transform(img1)
                img2 = transform(img2)
            elif isinstance(transform, transforms.RandomHorizontalFlip):
                if random.random() < 0.5:
                    img1 = transforms.functional.hflip(img1)
                    img2 = transforms.functional.hflip(img2)
            elif isinstance(transform, transforms.RandomCrop):
                i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=transform.size)
                img1 = transforms.functional.crop(img1, i, j, h, w)
                img2 = transforms.functional.crop(img2, i, j, h, w)
            elif isinstance(transform, transforms.ToTensor):
                img1 = transforms.functional.to_tensor(img1)
                img2 = transforms.functional.to_tensor(img2)
            elif callable(transform):
                img1 = transform(img1)
                img2 = transform(img2)
            else:
                raise NotImplementedError(f"Transform {transform} is not implemented in PairedTransform.")
        return img1, img2