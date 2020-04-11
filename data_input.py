import xmltodict
import pandas as pd
from pathlib import Path
import glob
from dataclasses import dataclass

@dataclass
class BoundingBox:
	xmin: int
	ymin: int
	xmax: int 
	ymax: int
	label: str

	def get_tuple(self):
		return (self.xmin, self.ymin, self.xmax, self.ymax, self.label)

def parse_xml_file_to_dict(image_xml_path):
	'''
	parse the given xml filepath into a dict

	image_xml_path: Path
	'''
	with open(image_xml_path, 'r') as f:
		lines = f.readlines()
		lines = ''.join(lines)

	xml_dict = xmltodict.parse(lines)
	return xml_dict

def parse_object(obj):
	'''
	obj: List[{}, ]
	obj represents the bounding box object
	generated via LabelImg and saved into xml format.
	'''

	bounding_box = obj['bndbox']
	xmin = bounding_box['xmin']
	ymin = bounding_box['ymin']
	xmax = bounding_box['xmax']
	ymax = bounding_box['ymax']
	label = obj['name']

	bb = BoundingBox(xmin=xmin, 
					ymin=ymin, 
					xmax=xmax, 
					ymax=ymax, 
					label=label)

	return bb

def parse_bounding_boxes_from_xml_dict(xml_dict):
	bounding_boxes = []
	objects = xml_dict['annotation']['object']
	
	# if there's only one bounding box, objects is a dict
	# instead of a list of dicts. handle that edge case here.
	# import pdb; pdb.set_trace()
	if not type(objects) is list:
		objects = [objects]

	for obj in objects:
		bb = parse_object(obj)
		bounding_boxes.append(bb)

	return bounding_boxes

training_data_dir = Path('/Users/anordin/python_auto_clicker/images')
training_image_paths = training_data_dir.glob('**/*.png')
columns = [
	"image_path",
	"xmin",
	"ymin",
	"xmax", 
	"ymax",
	"label"
]
rows = []
for training_image_path in training_image_paths:
	
	image_xml_path = training_image_path.with_suffix('.xml')
	print(f"Parsing {image_xml_path}")

	xml_dict = parse_xml_file_to_dict(image_xml_path)
	
	bounding_boxes = parse_bounding_boxes_from_xml_dict(xml_dict)

	for bounding_box in bounding_boxes:
		row = tuple([training_image_path]) +  bounding_box.get_tuple()
		rows.append(row)

df = pd.DataFrame(rows, columns=columns)

df.to_csv('training_data.csv', index=False, header=False)
