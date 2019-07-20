from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class BoardCocoDataset(CocoDataset):

    CLASSES = ('节')