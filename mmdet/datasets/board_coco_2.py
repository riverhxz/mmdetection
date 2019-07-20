from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class BoardJieDataset(CocoDataset):
    CLASSES = ('节')