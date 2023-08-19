# directory containing frames
frames_dir = './data/frames'
# directory containing labels and annotations
info_dir = './data/information'

# i3d model pretrained on Kinetics, https://github.com/yaohungt/Gated-Spatio-Temporal-Energy-Graph
i3d_pretrained_path = './rgb_i3d_pretrained.pt'

# num of frames in a single video
num_frames =196



# input data dims;
C, H, W = 3,224,224
# image resizing dims;
input_resize = 455, 256

# statistics of dataset
label_max = 104.5
label_min = 0.
judge_max = 10.
judge_min = 0.

# output dimension of I3D backbone
feature_dim = 1024

output_dim = {'TAAM': 21}

# num of judges
num_judges = 5
