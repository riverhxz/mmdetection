import xml.etree.ElementTree
import sys
import os

import tqdm
import cv2
import numpy as np
import pickle
classes = ["hole"]
class2id = dict(zip(classes, range(len(classes))))


def parse_annotation(path):
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
            left, top, right, bottom, name =  line.strip().split()

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
                'width': img.shape[1],
                'height': img.shape[0],
                'ann': {
                    'bboxes': box_class[:, :4],
                    'labels': box_class[:, 4],
                    # 'bboxes_ignore': np.empty([0]),
                }
            }

def main(argv):
    input, output_fn = argv

    from glob import glob
    collector = []
    for breed_dir in os.listdir(input):
        print(breed_dir)
        for annotation_file in tqdm.tqdm(glob(os.path.join(input, breed_dir, "*.txt"))):
            annotation = parse_annotation(annotation_file)
            collector.append(annotation)
    with open(output_fn, "wb") as output_file:
        output_file.write(pickle.dumps(collector))

if __name__ == '__main__':
        main(sys.argv[1:])
