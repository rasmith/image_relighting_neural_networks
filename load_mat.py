import numpy as np
import sys
import scipy.misc
import os

from ltm_renderer import LtmRenderer
from subprocess import call
from multiprocessing import Pool

dir_name = sys.argv[1]

print("dir_name = %s" % dir_name)

def render_matrix_to_channels(mat_file_name, output_dir):
  light_height = 32
  light_width = 32
  renderer = LtmRenderer()
  renderer.load(mat_file_name)
  os.mkdir(output_dir)
  rgb = np.zeros( (renderer.image_height, renderer.image_width, 3), \
      dtype = np.uint8)
  for i in range(0, 1024):
    output_file_name = '%04d' % i + '.png'
    light = np.zeros((1, light_height * light_width))
    light[0, i] = 1
    gray = renderer.render_from_light(light)
    rgb[:, :, 0] = gray
    rgb[:, :, 1] = gray
    rgb[:, :, 2] = gray
    scipy.misc.imsave(output_dir + '/' +output_file_name, rgb)


def combine_with_image_magick(arg):
  red_img, green_img, blue_img, rgb_img = arg
  print("red = %s,green = %s, blue = %s, rgb = %s" % \
      (red_img, green_img, blue_img, rgb_img))
  call(["convert", red_img, green_img, blue_img , '-combine' , rgb_img])

def combine_channels_with_image_magick(red_dir, green_dir, blue_dir, rgb_dir):
  os.mkdir(rgb_dir)
  pool = Pool(processes=8)  
  work_list = []
  for i in range(0, 1024):
    suffix = '%04d' % i + '.png'
    work_list.append((red_dir + '/' + suffix,
      green_dir +'/' + suffix, blue_dir + '/' + suffix,
      rgb_dir + '/' + suffix))
  pool.map(combine_with_image_magick, work_list)

def render_image(red_mat_file, green_mat_file, blue_mat_file, output_name, row):
  light_height = 32
  light_width = 32
  light = np.zeros((1, light_height * light_width))
  light[0, row] = 1
  renderer = LtmRenderer()
  rgb = np.zeros( (renderer.image_height, renderer.image_width, 3),  \
      dtype = np.uint8)
  renderer.load(red_mat_file)
  rgb[:, :, 0]  = renderer.render_from_light(light)
  renderer.load(green_mat_file)
  rgb[:, :, 1]  = renderer.render_from_light(light)
  renderer.load(blue_mat_file)
  rgb[:, :, 2]  = renderer.render_from_light(light)
  scipy.misc.imsave(output_name, rgb)


if sys.argv[2] == 'single':
  row = int(sys.argv[3])
  print('Single shot mode.')
  red_mat_file = dir_name + '/Red.mat'
  green_mat_file = dir_name + '/Green.mat'
  blue_mat_file = dir_name + '/Blue.mat'
  render_image(red_mat_file, green_mat_file, blue_mat_file, 'image.png', row)
elif sys.argv[2] == 'batch':
  print('Batch mode.')
  combine_channels_with_image_magick('r', 'g', 'b', 'rgb')

