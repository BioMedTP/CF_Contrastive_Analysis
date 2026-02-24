from abc import abstractmethod
import torchvision.transforms as transforms
from datasets import augmentations


class TransformsConfig(object):

	def __init__(self, opts):
		self.opts = opts

	@abstractmethod
	def get_transforms(self):
		pass


class EncodeTransforms(TransformsConfig):

    def __init__(self, opts):
        super().__init__(opts)

    def _make_transform(self, resolution, num_channels):
        """Creates a transform for given resolution and number of channels."""

        if num_channels == 3:
            mean = [0.5, 0.5, 0.5]
            std  = [0.5, 0.5, 0.5]
        else:
            mean = [0.5]
            std  = [0.5]

        if self.opts.random_flip:
            return transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
	
    def get_transforms(self):

        # Build dictionary dynamically
        transforms_dict = {}

        # Add RGB transforms
        for res in [256, 512, 1024]:
            transforms_dict[f"rgb{res}"] = self._make_transform(res, num_channels=3)

        # Add Grayscale transforms
        for res in [256, 512, 1024]:
            transforms_dict[f"grayscale{res}"] = self._make_transform(res, num_channels=1)

        return transforms_dict

# class EncodeTransforms(TransformsConfig):

# 	def __init__(self, opts):
# 		super(EncodeTransforms, self).__init__(opts)
# 		self.opts = opts
# 	def get_transforms(self):
# 		transforms_dict = {
# 			'transform_train': transforms.Compose([
# 				transforms.Resize((self.opts.transform_size, self.opts.transform_size)),
# 				transforms.RandomHorizontalFlip(0.5),
# 				transforms.ToTensor(),
# 				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
# 			'transform_source': None,
# 			'transform_test': transforms.Compose([
# 				transforms.Resize((self.opts.transform_size, self.opts.transform_size)),
# 				transforms.ToTensor(),
# 				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
# 			'transform_inference': transforms.Compose([
# 				transforms.Resize((self.opts.transform_size, self.opts.transform_size)),
# 				transforms.ToTensor(),
# 				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# 		}
# 		return transforms_dict


class FrontalizationTransforms(TransformsConfig):

	def __init__(self, opts):
		super(FrontalizationTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict


class SketchToImageTransforms(TransformsConfig):

	def __init__(self, opts):
		super(SketchToImageTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor()]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor()]),
		}
		return transforms_dict


class SegToImageTransforms(TransformsConfig):

	def __init__(self, opts):
		super(SegToImageTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.ToOneHot(self.opts.label_nc),
				transforms.ToTensor()]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.ToOneHot(self.opts.label_nc),
				transforms.ToTensor()])
		}
		return transforms_dict


class SuperResTransforms(TransformsConfig):

	def __init__(self, opts):
		super(SuperResTransforms, self).__init__(opts)

	def get_transforms(self):
		if self.opts.resize_factors is None:
			self.opts.resize_factors = '1,2,4,8,16,32'
		factors = [int(f) for f in self.opts.resize_factors.split(",")]
		print("Performing down-sampling with factors: {}".format(factors))
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.BilinearResize(factors=factors),
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.BilinearResize(factors=factors),
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict
