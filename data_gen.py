import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os

class_list = np.array(["aeroplane", "bicycle","bird","boat","bottle","bus","car",\
		"cat","chair","cow","diningtable","dog","horse", "motorbike",\
		"person","pottedplant","sheep","sofa","train","tvmonitor"])

train_list = np.loadtxt("VOCdevkit/VOC2012/ImageSets/Main/trainval.txt", dtype="str")


def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


## Data augmentation
def init_data_gen():
	global orig_nb_images, nb_images, nb_test, max_nb_obj_per_image, image_size
	global rot_amp, contrast_amp, flip_hor, flip_vert, shift_amp_w, shift_amp_h, zoom_prop
	global input_data, targets, input_test, targets_test, all_im, all_im_prop

	orig_nb_images = 11540
	nb_images = 2000
	nb_test = 100
	max_nb_obj_per_image = 48
	image_size = 256
	
	rot_amp = 0.0 #in deg, not yet in use
	contrast_amp = 0.0 #in percent
	flip_hor = 0.5 #total proportion
	flip_vert = 0.0 #total proportion
	shift_amp_w = 0.3 #percent of image_size
	shift_amp_h = 0.3 #percent of image_size
	zoom_prop = 0.0 #percent of imagesize

	if(not os.path.exists('./all_im.dat')):
		all_im = np.zeros((orig_nb_images, image_size, image_size, 3), dtype="uint8")
		all_im_prop = np.zeros((orig_nb_images, 4), dtype="float32")

		for i in tqdm(range(0, orig_nb_images)):

			im = Image.open("VOCdevkit/VOC2012/JPEGImages/"+train_list[i]+".jpg")
			width, height = im.size

			im = make_square(im)
			width2, height2 = im.size

			x_offset = int((width2 - width)*0.5)
			y_offset = int((height2 - height)*0.5)

			all_im_prop[i] = [x_offset, y_offset, width2, height2]

			im = im.resize((image_size,image_size))
			im_array = np.asarray(im)
			for depth in range(0,3):
				all_im[i,:,:,depth] = im_array[:,:,depth]

		all_im.tofile("all_im.dat")
		all_im_prop.tofile("all_im_prop.dat")
	else:
		all_im = np.fromfile("all_im.dat", dtype="uint8")
		all_im_prop = np.fromfile("all_im_prop.dat", dtype="float32")
		all_im = np.reshape(all_im, ((orig_nb_images, image_size, image_size, 3)))
		all_im_prop = np.reshape(all_im_prop,(orig_nb_images, 4))

	input_data = np.zeros((nb_images,image_size*image_size*3), dtype="float32")
	targets = np.zeros((nb_images,1+max_nb_obj_per_image*7), dtype="float32")

	input_test = np.zeros((nb_test,image_size*image_size*3), dtype="float32")
	targets_test = np.zeros((nb_test,1+max_nb_obj_per_image*7), dtype="float32")

## Data augmentation
def create_train_batch():

	for i in range(0, nb_images):
		
		i_d = np.random.randint(0,orig_nb_images)
		
		tree = ET.parse("VOCdevkit/VOC2012/Annotations/"+train_list[i_d]+".xml")
		root = tree.getroot()

		patch = np.copy(all_im[i_d])

		flip_w = 0
		flip_h = 0
		if(np.random.random() < flip_hor):
			flip_w = 1
			patch = np.flip(patch,axis=1)
		if(np.random.random() < flip_vert):
			flip_h = 1
			patch = np.flip(patch,axis=0)
			
		shift_w_val = (np.random.random()*2.0 - 1.0)*shift_amp_w
		shift_h_val = (np.random.random()*2.0 - 1.0)*shift_amp_h
		
		x_offset, y_offset, width2, height2 = all_im_prop[i_d]

		patch = roll_zeropad(patch, int(shift_w_val*image_size), axis=1)
		patch = roll_zeropad(patch, int(shift_h_val*image_size), axis=0)
		
		for depth in range(0,3):
			input_data[i,depth*image_size*image_size:(depth+1)*image_size*image_size] = patch[:,:,depth].flatten("C")/255.0

		k = 0
		obj_list = root.findall("object", namespaces=None)
		targets[i,0] = len(obj_list)
		for obj in obj_list:
			diff = obj.find("difficult", namespaces=None)
			if(diff.text == "1"):
				targets[i,0] -= 1
				continue
			oclass = obj.find("name", namespaces=None)
			bndbox = obj.find("bndbox", namespaces=None)
			
			int_class = np.where(class_list[:] == oclass.text)
			xmin = (flip_w*(image_size-1.0) + np.sign(0.5-flip_w)*int(float(bndbox.find("xmin").text)+x_offset)*image_size/width2) + shift_w_val*image_size
			ymin = (flip_h*(image_size-1.0) + np.sign(0.5-flip_h)*int(float(bndbox.find("ymin").text)+y_offset)*image_size/height2) + shift_h_val*image_size
			xmax = (flip_w*(image_size-1.0) + np.sign(0.5-flip_w)*int(float(bndbox.find("xmax").text)+x_offset)*image_size/width2) + shift_w_val*image_size
			ymax = (flip_h*(image_size-1.0) + np.sign(0.5-flip_h)*int(float(bndbox.find("ymax").text)+y_offset)*image_size/height2) + shift_h_val*image_size
			
			if(xmax < xmin):
				temp = xmin
				xmin = xmax
				xmax = temp
			if(ymax < ymin):
				temp = ymin
				ymin = ymax
				ymax = temp
			
			box_cx = (xmax+xmin)*0.5
			box_cy = (ymax+ymin)*0.5
			
			if(box_cx < 0 or box_cy < 0 or box_cx > image_size or box_cy > image_size):
				targets[i,0] -= 1
				continue
			xmin = max(0, xmin)
			ymin = max(0, ymin)
			xmax = min(image_size, xmax)
			ymax = min(image_size, ymax)
			
			targets[i,1+k*7:1+(k+1)*7] = np.array([int_class[0][0]+1, xmin,ymin,0.0,xmax,ymax,1.0])
			k += 1
		

	return input_data, targets

def create_test_batch():

	for i in range(0, nb_test):
		
		i_d = i*10

		tree = ET.parse("VOCdevkit/VOC2012/Annotations/"+train_list[i_d]+".xml")
		root = tree.getroot()
		
		patch = np.copy(all_im[i_d])

		x_offset, y_offset, width2, height2 = all_im_prop[i_d]

		for depth in range(0,3):
			input_test[i,depth*image_size*image_size:(depth+1)*image_size*image_size] = patch[:,:,depth].flatten("C")/255.0
		
		k = 0
		obj_list = root.findall("object", namespaces=None)
		targets_test[i,0] = len(obj_list)
		for obj in obj_list:
			diff = obj.find("difficult", namespaces=None)
			if(diff.text == "1"):
				targets_test[i,0] -= 1
				continue
			oclass = obj.find("name", namespaces=None)
			bndbox = obj.find("bndbox", namespaces=None)

			int_class = np.where(class_list[:] == oclass.text)
			xmin = int(float(bndbox.find("xmin").text)+x_offset)*image_size/width2
			ymin = int(float(bndbox.find("ymin").text)+y_offset)*image_size/height2
			xmax = int(float(bndbox.find("xmax").text)+x_offset)*image_size/width2
			ymax = int(float(bndbox.find("ymax").text)+y_offset)*image_size/height2

			if(xmax < xmin):
				temp = xmin
				xmin = xmax
				xmax = temp
			if(ymax < ymin):
				temp = ymin
				ymin = ymax
				ymax = temp
			
			box_cx = (xmax+xmin)*0.5
			box_cy = (ymax+ymin)*0.5
			
			if(box_cx < 0 or box_cy < 0 or box_cx > image_size or box_cy > image_size):
				targets_test[i,0] -= 1
				continue
			xmin = max(0, xmin)
			ymin = max(0, ymin)
			xmax = min(image_size, xmax)
			ymax = min(image_size, ymax)

			targets_test[i,1+k*7:1+(k+1)*7] = np.array([int_class[0][0]+1, xmin,ymin,0.0,xmax,ymax,1.0])
			k += 1
	
	return input_test, targets_test


