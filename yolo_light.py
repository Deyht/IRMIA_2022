import numpy as np
from threading import Thread
import data_gen as gn

import sys
sys.path.insert(0,'/content/CIANNA/src/build/lib.linux-x86_64-3.7')
import CIANNA as cnn


def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

def data_augm():
	input_data, targets = gn.create_train_batch()
	cnn.delete_dataset("TRAIN_buf", silent=1)
	cnn.create_dataset("TRAIN_buf", nb_images, input_data[:,:], targets[:,:], silent=1)
	return

nb_images = 2000
nb_test = 100
nb_param = 0

max_nb_obj_per_image = 48

im_size = 256
nb_box = 6


load_epoch = 0
if (len(sys.argv) > 1):
	load_epoch = int(sys.argv[1])


cnn.init(in_dim=i_ar([im_size,im_size]), in_nb_ch=3, out_dim=1+max_nb_obj_per_image*(7+nb_param),
	 b_size=16, comp_meth='C_CUDA', dynamic_load=1, mixed_precision="FP16C_FP32A")

gn.init_data_gen()

input_data, targets = gn.create_train_batch()
input_test, targets_test = gn.create_test_batch()

cnn.create_dataset("TRAIN", nb_images, input_data[:,:], targets[:,:])
cnn.create_dataset("VALID", nb_test, input_test[:,:], targets_test[:,:])
cnn.create_dataset("TEST" , nb_test, input_test[:,:], targets_test[:,:])

##### YOLO parameters tuning #####

#Size priors for all possible boxes per grid. element 
priors_w = f_ar([20.,40.,60.,80.,120.,180.])
priors_h = f_ar([20.,40.,60.,80.,120.,180.])
priors_d = f_ar([1.0,1.0,1.0,1.0,1.0,1.0])
#No obj probability prior to rebalance the size distribution
prior_noobj_prob = f_ar([0.1,0.1,0.1,0.05,0.05,0.05])

#Relative scaling of each error "type" : 
#[Position, Size, Probability, Objectness, Class, Ex. Param]
error_scales = f_ar([2.0, 3.0, 1.0, 5.0, 2.0, 1.0])
#Relative scaling of each extra paramater
param_ind_scales = f_ar([1.0])
#Various IoU limit conditions
#[Good but not best boxes, Prob. fit, Obj. fit, class fit, param fit] 
IoU_limits = f_ar([0.2, 0.0, 0.3, 0.0, 0.3])
#Activate / deactivate some parts of the loss
#[Position, Size, Probability & Objectness, Classification, Ex. Param]
fit_parts = i_ar([1, 1, 1, 1, 0])
#Supplementary parameters for activation function of each part
#[[Postion (sigmoid): beta , exp(lim), None   ]
# [Size    (linear) : slope, max lim , min lim]
# [Proba   (sigmoid): beta , exp(lim), None   ]
# [Object. (sigmoid): beta , exp(lim), None   ]
# [Class.  (sigmoid): beta , exp(lim), None   ]
# [Param.  (linear) : slope, max lim , min lim]]
slopes_and_maxes = f_ar([[2.0, 8.0, 0.0],\
			 [0.5, 1.8, -1.4],\
			 [1.0, 8.0, 0.0],\
			 [1.0, 8.0, 0.0],\
			 [2.0, 8.0, 0.0],\
			 [1.0, 2.0, -0.2]])

nb_yolo_filters = cnn.set_yolo_params(nb_box=nb_box, prior_w=priors_w, prior_h=priors_h, prior_d = priors_d,\
						prior_noobj_prob=prior_noobj_prob, nb_class=20, nb_param=nb_param, IoU_type = "DIoU",
						error_scales = error_scales, param_ind_scales = param_ind_scales, slopes_and_maxes = slopes_and_maxes,
						IoU_limits = IoU_limits, fit_parts = fit_parts, strict_box_size_association = 1)

relu_a = cnn.relu(saturation=200.0, leaking=0.2)

if(load_epoch > 0):
	cnn.load("net_save/net0_s%04d.dat"%load_epoch,load_epoch)
else:
	
	cnn.conv(f_size=i_ar([3,3]), nb_filters=16  , padding=i_ar([1,1]), activation=relu_a)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	cnn.conv(f_size=i_ar([3,3]), nb_filters=32  , padding=i_ar([1,1]), activation=relu_a)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	cnn.conv(f_size=i_ar([3,3]), nb_filters=64 , padding=i_ar([1,1]), activation=relu_a)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	cnn.conv(f_size=i_ar([3,3]), nb_filters=128 , padding=i_ar([1,1]), activation=relu_a)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	cnn.conv(f_size=i_ar([3,3]), nb_filters=128 , padding=i_ar([1,1]), activation=relu_a)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	cnn.conv(f_size=i_ar([3,3]), nb_filters=256, padding=i_ar([1,1]), activation=relu_a)

	cnn.conv(f_size=i_ar([1,1]), nb_filters=nb_yolo_filters, padding=i_ar([0,0]), activation="YOLO")
	
for block in range(0,2000): #2000 

	if(block == 0 or np.mod(block,50) == 0):
		cnn.forward(drop_mode="AVG_MODEL", saving = 0, no_error=2)
		
	t = Thread(target=data_augm)
	t.start()
	
	cnn.train(nb_epoch=1, learning_rate=0.0001, end_learning_rate=0.000001, shuffle_every=0,\
			 momentum=0.5, decay=0.0005, save_every=50, silent=1, TC_scale_factor=32.0)
				 
	if(block == 0):
		cnn.perf_eval()

	t.join()
	
	cnn.swap_data_buffers("TRAIN")

t.stop()
exit()













