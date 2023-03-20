import json
import os
import shutil
import random

class_map = {'Larb': 0, 'Pidan': 1, 'Balut': 2, 'Souffle': 3}

with open('data/raw/raw_label.json', 'r') as f:
    raw = json.load(f)

for item in raw:
    f_name = item['file_upload'].split('.')[0] + '.txt'
    boxes = []

    for box in item['annotations'][0]['result']:
        x = box['value']['x']
        y = box['value']['y']
        assert x >= 0
        assert y >= 0
        width = box['value']['width']
        height = box['value']['height']
        width = min(width, 100 - x)
        height = min(height, 100 - y)
        assert x + width <= 100
        assert y + height <= 100
        duck = class_map[box['value']['rectanglelabels'][0]]
        boxes.append(
            ' '.join(map(str, [duck, (x + width / 2) / 100, (y + height / 2) / 100, width / 100, height / 100])))

    with open(f'data/all_labels/{f_name}', 'w') as f:
        f.write('\n'.join(boxes))

shutil.rmtree('data/images')
shutil.rmtree('data/labels')
all_images = os.listdir('data/all_images')
random.seed(111)
random.shuffle(all_images)

os.makedirs('data/images/val')
os.makedirs('data/labels/val')
os.makedirs('data/images/train')
os.makedirs('data/labels/train')

for img in all_images[:30]:
    label = img.split('.')[0] + '.txt'
    shutil.copy(os.path.join('data/all_images', img), os.path.join('data/images/val', img))
    shutil.copy(os.path.join('data/all_labels', label), os.path.join('data/labels/val', label))

for img in all_images[30:]:
    label = img.split('.')[0] + '.txt'
    shutil.copy(os.path.join('data/all_images', img), os.path.join('data/images/train', img))
    shutil.copy(os.path.join('data/all_labels', label), os.path.join('data/labels/train', label))