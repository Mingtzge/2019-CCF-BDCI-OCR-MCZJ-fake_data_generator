import no_watermark
import random
import os
from multiprocessing import Pool
import cv2

from PIL import Image, ImageEnhance, ImageStat, ImageFilter
import os
import numpy as np


template_file='roi.jpg'	# 水印的模板
demo_file='roi_2.jpg'
def run(img_name):
    try:
        image_f = os.path.join(origin_path, img_name)
        save_img = os.path.join(dst_path, img_name)
        file = files[random.randint(0, len(files) - 1)]
        image_file = file_path + '/' + file                                     # 0.5参数不用管，我好像没用到
        m = no_watermark.WatermarkBlender(image_file, template_file, demo_file, 0.5) # demo_file这个参数不用管，这个是上一个类中需要用到的内容，只需要在文件夹下有这样一>个文件就可以了
        m.add_watermark(image_file, image_f, save_img)
    except:
        print(img_name, "  failed!!")


if __name__=='__main__':

    file_path ="rematch_jinzhifuyin_train_empty"
    origin_path = "fake_real_color_data_jinzhifuyin"
    dst_path = "../fake_originImages_add_wartermask"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    files = os.listdir(file_path)
    origin_files = os.listdir(origin_path)
    for origin_file in origin_files:
        run(origin_file)
    # pool = Pool(30)
    #
    # pool.map(run, origin_files[:300000])
    # pool.close()
    # pool.join()
