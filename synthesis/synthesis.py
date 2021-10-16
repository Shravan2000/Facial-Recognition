
from argparse import ArgumentParser
from glob import glob
from PIL import Image
import os
import sys
from PIL import Image
import numpy as np
import math
import scipy.io as sio
from skimage import io
from time import time
import subprocess
import cv2

sys.path.append('..')
import face3d
from face3d import mesh

parser = ArgumentParser()
parser.add_argument('-i', type=str, help='Directory where 3D images are kept.')
parser.add_argument('-o', type=str, help='Directory where to output images.')


def transform_test(vertices, obj, camera, h=256, w=256):
    R = mesh.transform.angle2matrix(obj['angles'])
    transformed_vertices = mesh.transform.similarity_transform(
        vertices, obj['s'], R, obj['t'])

    if camera['proj_type'] == 'orthographic':
        projected_vertices = transformed_vertices
        image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    else:

        camera_vertices = mesh.transform.lookat_camera(
            transformed_vertices, camera['eye'], camera['at'], camera['up'])
        projected_vertices = mesh.transform.perspective_project(
            camera_vertices, camera['fovy'], near=camera['near'], far=camera['far'])
        image_vertices = mesh.transform.to_image(
            projected_vertices, h, w, True)

    rendering = mesh.render.render_colors(
        image_vertices, triangles, colors, h, w)
    rendering = np.minimum((np.maximum(rendering, 0)), 1)
    return rendering

def main():
    args = parser.parse_args()
    image_folder = args.i
    save_folder = args.o
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    types = ('*.jpg', '*.png')
    image_paths = []
    for files in types:
        image_paths.extend(glob(os.path.join(image_folder, files)))

    for image_path in image_paths:
        name = image_path.strip().split('/')[-1][:-4]
        C = sio.loadmat(name + '_mesh.mat')
        vertices = C['vertices']
        global colors
        global triangles
        colors = C['colors']
        triangles = C['triangles']
        colors = colors/np.max(colors)
        vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]
        obj = {}
        camera = {}
        scale_init = 180/(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))

        camera['proj_type'] = 'orthographic'
        factor = 1.0
        obj['s'] = scale_init*factor
        obj['angles'] = [0, 0, 0]
        obj['t'] = [0, 0, 0]
        image = transform_test(vertices, obj, camera)
        io.imsave(os.path.join(save_folder, name + '.jpg'), image)
                
        angle = 30
        obj['s'] = scale_init
        obj['angles'] = [0, 0, 0]
        obj['angles'][0] = angle
        obj['t'] = [0, 0, 0]
        image = transform_test(vertices, obj, camera)
        io.imsave(os.path.join(save_folder, name + '_0_30.jpg'), image)
        
        angle = -30
        obj['s'] = scale_init
        obj['angles'] = [0, 0, 0]
        obj['angles'][0] = angle
        obj['t'] = [0, 0, 0]
        image = transform_test(vertices, obj, camera)
        io.imsave(os.path.join(save_folder, name + '_0_-30.jpg'), image)

        angle = 20
        obj['s'] = scale_init
        obj['angles'] = [0, 0, 0]
        obj['angles'][2] = angle
        obj['t'] = [0, 0, 0]
        image = transform_test(vertices, obj, camera)
        io.imsave(os.path.join(save_folder, name + '_2_20.jpg'), image)

        angle = -20
        obj['s'] = scale_init
        obj['angles'] = [0, 0, 0]
        obj['angles'][2] = angle
        obj['t'] = [0, 0, 0]
        image = transform_test(vertices, obj, camera)
        io.imsave(os.path.join(save_folder, name + '_2_-20.jpg'), image)

        angle = 30
        obj['s'] = scale_init
        obj['angles'] = [0, 0, 0]
        obj['angles'][1] = angle
        obj['t'] = [0, 0, 0]
        image = transform_test(vertices, obj, camera)
        io.imsave(os.path.join(save_folder, name + '_1_30.jpg'), image)

        angle = -30
        obj['s'] = scale_init
        obj['angles'] = [0, 0, 0]
        obj['angles'][1] = angle
        obj['t'] = [0, 0, 0]
        image = transform_test(vertices, obj, camera)
        io.imsave(os.path.join(save_folder, name + '_1_-30.jpg'), image)
   
    types = ('*.jpg', '*.png')
    image_paths= []
    for files in types:
        image_paths.extend(glob(os.path.join(save_folder+'/input/*', files)))
    for image_path in image_paths:
        name = image_path.strip().split('/')[-1][:-4]
        img = cv2.imread(image_path,1)
        img_rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(os.path.join(save_folder, name + '.jpg'), img_rotate_180)
    
if __name__ == '__main__':
    main()
    

