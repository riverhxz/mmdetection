import xml.etree.ElementTree
import sys
import os

import tqdm
import cv2
import numpy as np
import pickle

classes = ["hole"]
class2id = dict(zip(classes, range(len(classes))))


def parse_annotation(path, imgid):

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
        fn = im_path.split("/")[-1]
        img = {
            "license": 1,
            "file_name": fn,
            "coco_url": f"http://mock/{fn}",
            "height": img.shape[0],
            "width": img.shape[1],
            "date_captured": "2013-11-14 17:02:52",
            "flickr_url": f"http://mock.com/{fn}",
            "id": imgid
        }

        box_class[:, 2:4] = box_class[:, 2:4] - box_class[:, 0:2]
        annotations = [
            {
                "segmentation": [[]],
                "area": int(box[2] * box[3]),
                "iscrowd": 0,
                "image_id": imgid,
                "bbox": box[:4].tolist(),
                "category_id": box[4].tolist(),
                "id": imgid * 100 + i
            }
            for i, box in enumerate(box_class)
        ]

        return img, annotations


def main(argv):
    """
    {
        "info": {...},
        "licenses": [...],
        "images": [...],
        "annotations": [...],
        "categories": [...], <-- Not in Captions annotations
        "segment_info": [...] <-- Only in Panoptic annotations
    }
    :param argv:
    :return:
    """
    input, output_fn = argv

    from glob import glob
    categories = [{"supercategory": "hole", "id": 0, "name": "hole"}, ]

    info = {
        "description": "WOODBOARD 2019 Dataset",
        "url": "http://dyh.org",
        "version": "1.0",
        "year": 2019,
        "contributor": "dyh,hhh",
        "date_created": "2019/05/01"
    }
    licenses = [
        {
            "url": "http://mock.licence.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "mock License"
        },
    ]
    annotations = []
    images = []

    for i,annotation_file in tqdm.tqdm(enumerate(glob(os.path.join(input, "*.txt")))):
        image, anno = parse_annotation(annotation_file,i)
        images.append(image)
        annotations.extend(anno)

    tmp_coco = {
        "info": info,
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    import json
    with open(output_fn, "w") as output_file:
        output_file.write(json.dumps(tmp_coco))


if __name__ == '__main__':
    main(sys.argv[1:])
