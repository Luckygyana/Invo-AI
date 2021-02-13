import sys
import json
from PIL import Image, ImageDraw
from enum import Enum
import pandas as pd
class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

"""
dict_keys(['cropHintsAnnotation', 'fullTextAnnotation', 'imagePropertiesAnnotation', 'labelAnnotations', 'safeSearchAnnotation', 'textAnnotations']) 
"""
# print(data['fullTextAnnotation']['pages'][0]['blocks'][1])

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5

def get_document_bounds(data, feature):
    # [START vision_document_text_tutorial_detect_bounds]
    """Returns document bounds given an image."""

    bounds = []



    # Collect specified feature bounds by enumerating all document features
    for page in data['fullTextAnnotation']['pages']:
        for block in page['blocks']:
            for paragraph in block['paragraphs']:
                for word in paragraph['words']:
                    for symbol in word['symbols']:
                        if (feature == FeatureType.SYMBOL):
                            bounds.append(symbol['boundingBox'])

                    if (feature == FeatureType.WORD):
                        bounds.append(word['boundingBox'])

                if (feature == FeatureType.PARA):
                    bounds.append(paragraph['boundingBox'])

            if (feature == FeatureType.BLOCK):
                print(block.keys())
                bounds.append(block['boundingBox'])

    # The list `bounds` contains the coordinates of the bounding boxes.
    # [END vision_document_text_tutorial_detect_bounds]
    return bounds

def draw_boxes(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        
        bound = Map(bound)
        print(bound.vertices)
        # sys.exit(0)
        draw.polygon([
            bound.vertices[0]['x'], bound.vertices[0]['y'],
            bound.vertices[1]['x'], bound.vertices[1]['y'],
            bound.vertices[2]['x'], bound.vertices[2]['y'],
            bound.vertices[3]['x'], bound.vertices[3]['y']], fill=(0,255,0,127) )
    return image

def bbox_csv(bounds, label, path):
    df = []
    for bound in bounds:
        bound = Map(bound)
        _dict = dict()
        _dict = {'xmin' : min([bound.vertices[0]['x'],
                               bound.vertices[1]['x'],
                               bound.vertices[2]['x'],
                               bound.vertices[3]['x']]),
                 'xmax' : max([bound.vertices[0]['x'],
                               bound.vertices[1]['x'],
                               bound.vertices[2]['x'],
                               bound.vertices[3]['x']]),
                 'ymin' : min([bound.vertices[0]['y'],
                               bound.vertices[1]['y'],
                               bound.vertices[2]['y'],
                               bound.vertices[3]['y']]),
                 'ymax' : max([bound.vertices[0]['y'],
                               bound.vertices[1]['y'],
                               bound.vertices[2]['y'],
                               bound.vertices[3]['y']]),
                 'label' : label

                }
        df.append(_dict)
    df = pd.DataFrame(df)
    df.to_csv(path.split('.')[0]+'.csv', index=False)



def render_doc_text(data, path, fileout):
    image = Image.open(path)
    block_bounds = get_document_bounds(data, FeatureType.BLOCK)
    draw_boxes(image, block_bounds, 'blue')
    para_bounds = get_document_bounds(data, FeatureType.PARA)
    # draw_boxes(image, para_bounds, 'red')
    # word_bounds = get_document_bounds(data, FeatureType.WORD)
    # print(para_bounds)
    # draw_boxes(image, word_bounds, 'yellow')
    bbox_csv(block_bounds, 'block', path)
    if fileout != 0:
        image.save(fileout)
    else:
        image.show()


def to_csv(bounds, label, path):
    df = []
    for bound in bounds:
        bound = Map(bound)
        _dict = dict()
        _dict = {'xmin' : min([bound.vertices[0]['x'],
                               bound.vertices[1]['x'],
                               bound.vertices[2]['x'],
                               bound.vertices[3]['x']]),
                 'xmax' : max([bound.vertices[0]['x'],
                               bound.vertices[1]['x'],
                               bound.vertices[2]['x'],
                               bound.vertices[3]['x']]),
                 'ymin' : min([bound.vertices[0]['y'],
                               bound.vertices[1]['y'],
                               bound.vertices[2]['y'],
                               bound.vertices[3]['y']]),
                 'ymax' : max([bound.vertices[0]['y'],
                               bound.vertices[1]['y'],
                               bound.vertices[2]['y'],
                               bound.vertices[3]['y']]),
                 'label' : label

                }
        df.append(_dict)
    df = pd.DataFrame(df)
    df.to_csv(path.split('.')+'.csv', index=False)


with open('src/table.json', 'r', encoding="utf-8") as f:
    data = json.load(f)
data = Map(data)
path = r'E:\source\sem_8\invoice\invoice\imgs\200\Sample1-0.jpg'
render_doc_text(data, path, fileout='temp.jpg')


