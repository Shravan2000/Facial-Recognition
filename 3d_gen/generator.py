import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast

from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture

def main(args):
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    prn = PRN(is_dlib = args.isDlib)

    
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)

    for i, image_path in enumerate(image_path_list):

        name = image_path.strip().split('/')[-1][:-4]

        
        image = imread(image_path)
        [h, w, c] = image.shape
        if c>3:
            image = image[:,:,:3]

        
        if args.isDlib:
            max_size = max(image.shape[0], image.shape[1])
            if max_size> 1000:
                image = rescale(image, 1000./max_size)
                image = (image*255).astype(np.uint8)
            pos = prn.process(image) 
        else:
            if image.shape[0] == image.shape[1]:
                image = resize(image, (256,256))
                pos = prn.net_forward(image/255.) 
            else:
                box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) 
                pos = prn.process(image, box)
        
        image = image/255.
        if pos is None:
            continue
        vertices = prn.get_vertices(pos)
        x='False'
        if x=='False':
            save_vertices = frontalize(vertices)
        else:
            save_vertices = vertices.copy()
        save_vertices[:,1] = h - 1 - save_vertices[:,1]
        colors = prn.get_colors(image, vertices)
        write_obj_with_colors(os.path.join(save_folder, name + '.obj'), vertices, prn.triangles, colors)
        sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='TestImages/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/results', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    main(parser.parse_args())
