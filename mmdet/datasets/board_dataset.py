from mmdet.datasets import CustomDataset

from .coco import CocoDataset
from .registry import DATASETS
import sys
import os

import tqdm
import cv2
import numpy as np
import pickle
classes = ["hole"]
class2id = dict(zip(classes, range(len(classes))))
train_percentile = 0.9
CLASSES = ('hole')
random_seed=10001
@DATASETS.register_module
class BoardDataset(CustomDataset):


    def getImgIds(self):
        return [f.split("/")[-1] for f in self.img_infos]

    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            anna = pickle.loads(f.read())
            np.random.seed(random_seed)
            np.random.shuffle(anna)
            return anna

@DATASETS.register_module
class BoardDatasetTrain(BoardDataset):
    def load_annotations(self, ann_file):
        anna = super(BoardDatasetTrain, self).load_annotations(ann_file)
        train = int(len(anna) * train_percentile)
        return anna[:train]

@DATASETS.register_module
class BoardDatasetTest(BoardDataset):
    def load_annotations(self, ann_file):
        anna = super(BoardDatasetTest, self).load_annotations(ann_file)
        train = int(len(anna) * train_percentile)
        return anna[train:]

def _parse_annotation(path):
    """

    :param path:
    :return:
          [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]
    """
    with open(path) as f:
        lines = f.readlines()
        num, holes = lines[0], lines[1:]
        cc = []

        for line in holes:
            left, top, right, bottom, name = line.strip().split()

            cc.append([
                int(left)
                , int(top)
                , int(right)
                , int(bottom)
                , class2id[name.lower()]])

        box_class = np.array(cc)
        im_path = path.replace(".txt", ".jpg")
        img = cv2.imread(im_path)
        img.shape[:2]
        return {
            'filename': im_path,
            'width': img[1],
            'height': img[0],
            'ann': {
                'bboxes': box_class[:, :4],
                'labels': box_class[:, 4],
                'bboxes_ignore': np.empty,
            }
        }

def main(argv):
    input, output_fn = argv

    annotations_root_dir = input
    from glob import glob
    collector = []
    for breed_dir in tqdm.tqdm(os.listdir(input)):
        print(breed_dir)
        for annotation_file in glob(os.path.join(input, breed_dir, "*.txt")):
            annotation = _parse_annotation(annotation_file)
            collector.append(annotation)
    with open(output_fn, "wb") as output_file:
        output_file.write(pickle.dumps(collector))

if __name__ == '__main__':
        main(sys.argv[1:])
