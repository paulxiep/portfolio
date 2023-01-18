import pyproj
import numpy as np
import json
import cv2
import shapefile

countries_json = 'states/countries.geo.json'
bioregion_shape = 'ecoregion/data/commondata/data0/wwf_terr_ecos.shp'

def states_to_d2(states, height=3000, width=4000):
    from detectron2.data.detection_utils import annotations_to_instances
    from detectron2.structures import Instances
    def flatten(polygon):
        return np.array(polygon).flatten()
    def minmax(coors):
        xs = [coor[0] for coor in coors]
        ys = [coor[1] for coor in coors]
        return [min(xs), min(ys), max(xs), max(ys)]

    if not states:
        return Instances((height, width))

    annotations = [
        {
            'segmentation': list(map(flatten, obj['polygon'])),
            'category_id': obj['category_id'],
#             'category_id': cls_dict[obj[task]],
            'bbox_mode': 0,
            'bbox': minmax(obj['polygon'][0][0])
        }
        for obj in states]
    # this method gives gt instead of pred (gt_classes instead of pred_classes, etc)
    output = annotations_to_instances(annotations, (height, width), 'bitmask')
#     output.scores = scores
    output.pred_masks = output.gt_masks.tensor.int()
#     output.pred_classes = output.gt_classes
    output.pred_boxes = output.gt_boxes

    # remove unused fields
    output.remove('gt_masks')
    output.remove('gt_classes')
    output.remove('gt_boxes')
    return output

def draw_image(states):
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data.catalog import Metadata
    states
    img = np.zeros([3000, 4000, 3]) + 127
    instances = states_to_d2(states)
    # metadata = Metadata()
    # metadata.thing_classes = {i: item['id'] for i, item in enumerate(states['features'])}
    v = Visualizer(img, instance_mode=1)
    if instances.has('pred_boxes'):
        instances.remove('pred_boxes')
    image = v.draw_instance_predictions(instances).get_image()
    return image

def coors_to_pixels(coors, from_latlon=True):
    def web_mercator_to_pixels(coor):
        x = max(min((coor[0]-limits['xmin'])*xmul, 4000), 0)
        y = max(min(3000-(coor[1]-limits['ymin'])*ymul, 3000), 0)
        assert 0 <= x <= 4000, x
        assert 0 <= y <= 3000, y
        return x, y
    def transform(coor):
        return web_mercator_to_pixels(transformer.transform(coor[1], coor[0]))
    if from_latlon:
        if len(np.array(coors[0]).shape) == 1:
            return [list(map(transform, coors))]
        else:
            return [list(map(transform, coor)) for coor in coors]
    else:
        return [list(map(web_mercator_to_pixels, coors))]

if __name__ == '__main__':
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857")
    with open(countries_json, 'r') as f:
        states = json.load(f)
    limits = {'xmin': -20037508.3427892,
                'ymin': -20037508.3427892,
                'xmax': 20037508.3427892,
                'ymax': 20037508.3427892}
    xmul = 4000/(limits['xmax']-limits['xmin'])
    ymul = 3000/(limits['ymax']-limits['ymin'])

    states = [{'polygon': [coors_to_pixels(coors) for coors in item['geometry']['coordinates']],
              'category_id': i}
                  for i, item in enumerate(states['features'])]

    image = draw_image(states)
    cv2.imwrite('countries.jpg', image)

    shape = shapefile.Reader(bioregion_shape, encoding="ISO-8859-1")
    ecos = {}
    for record in shape.shapeRecords():
        ecos[record.record[3]] = ecos.get(record.record[3], []) + [coors_to_pixels(record.shape.points, False)]
    n_ecos = len(ecos.keys())
    ecos = [{'category_id': i, 'polygon': value} for i, value in zip(range(180, 180 + n_ecos), ecos.values())]
    image = draw_image(ecos)
    cv2.imwrite('ecos.jpg', image)
