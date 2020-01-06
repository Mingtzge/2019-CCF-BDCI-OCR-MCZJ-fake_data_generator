#! encoding:utf-8
from multiprocessing import Pool
import os
import numpy as np
import random
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageStat, ImageEnhance, ImageFilter  # 导入模块
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

range_thr = {'12210-11983': 0.0,
             '11983-11899': 0.1,
             '11899-10881': 0.2,
             '10881-9006': 0.3,
             '9006-6943': 0.4,
             '6943-4996': 0.5,
             '4996-3427': 0.6,
             '3427-2239': 0.7,
             '2239-1435': 0.8,
             '1435-894': 0.9,
             '894-566': 1.0,
             '566-372': 1.1,
             '372-251': 1.2,
             '251-175': 1.3,
             '175-125': 1.4,
             '125-99': 1.5,
             '99-79': 1.6,
             '79-63': 1.7}


def getImageVar(imgPath):
    image = cv2.imread(imgPath)
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imageVar


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


def add_img_to_image(image, roi_img, b_thr, p, Gaussian_thr):
    new_img = Image.new('RGBA', (image.size[0], image.size[1]), (0, 0, 0, 0))
    new_img.paste(image)
    rgba_image = new_img.convert('RGBA')
    text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
    image_draw = ImageDraw.Draw(text_overlay)
    bright_random = min(random.randint(b_thr + 10, b_thr + 25), 190)
    x = max(int(np.random.normal(10, 5)), 0)
    y = max(int(np.random.normal(10, 5)), 0)
    if x > width - roi_img.size[0] - 1:
        x = width - roi_img.size[0] - 1
    if y > height - roi_img.size[1] - 1:
        y = height - roi_img.size[1] - 1
    p_new = (p[0] + x, p[1] + y)
    for i in range(roi_img.size[0]):
        for j in range(roi_img.size[1]):
            roi_pixel = roi_img.getpixel((i, j))
            if i in range(10, roi_img.size[0] - 10) and j in range(10, roi_img.size[1] - 10):
                if roi_pixel[0] > 150:
                    image_draw.point((i + p_new[0], j + p_new[1]), fill=(
                        max(0, int(roi_pixel[0] - bright_random / 5)),
                        max(0, int(roi_pixel[1] - bright_random / 5)),
                        max(0, int(roi_pixel[2] - bright_random / 5)), bright_random - int(Gaussian_thr * 10)))
                else:
                    image_draw.point((i + p_new[0], j + p_new[1]), fill=(
                        max(0, int(roi_pixel[0] - bright_random / 3)),
                        max(0, int(roi_pixel[1] - bright_random / 3)),
                        max(0, int(roi_pixel[2] - bright_random / 3)), bright_random - int(Gaussian_thr * 10)))
            elif roi_pixel[0] < 195:
                image_draw.point((i + p_new[0], j + p_new[1]), fill=(
                    max(0, int(roi_pixel[0] - bright_random / 3)),
                    max(0, int(roi_pixel[1] - bright_random / 3)),
                    max(0, int(roi_pixel[2] - bright_random / 3)), bright_random - int(Gaussian_thr * 10)))
    # image_merge = Image.blend(rgba_image, text_overlay, 0.2)
    # enh_sha = ImageEnhance.Sharpness(text_overlay)
    # sharpness = 100
    # text_overlay = enh_sha.enhance(sharpness)
    enh_sha = ImageEnhance.Sharpness(text_overlay)
    sharpness = bright_random / 80
    text_overlay = enh_sha.enhance(sharpness)
    # text_over = new_.filter(ImageFilter.CONTOUR)
    # text_overlay = Image.alpha_composite(text_over, text_overlay)
    # text_overlay = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0)(text_overlay)
    text_overlay = text_overlay.filter(MyGaussianBlur(radius=min(0, Gaussian_thr - 0.2)))
    image_merge = Image.alpha_composite(rgba_image, text_overlay)
    # 裁切图片
    image_with_text = image_merge.crop((p[0], p[1], p[0] + width, p[1] + height))
    # text_over.show()
    return image_with_text


def judge_in_area(h, w, x, y, pt):
    if ((x in range(pt[0], pt[0] + width)) and (y in range(pt[1], pt[1] + height))) or x > w or y > h:
        return False
    return True

    # 将图像转化为二值化图像


def transfer_bin_img(img_path):
    threshold = random.randint(130, 210)
    img = Image.open(img_path)

    # 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
    Img = img.convert('L')
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    photo = Img.point(table, '1')
    return photo


def gen_run(img_n):
    print(img_n)
    img_path = os.path.join(choose_img_path, img_n)
    img_rgb = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(roi_img_path, 0)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(res)
    origin_img = Image.open(img_path)
    var = getImageVar(img_path)
    min_ = 63
    for range_t in range_thr:
        items = range_t.split("-")
        max_ = int(items[0]) + 1
        min_ = int(items[1]) - 1
        if int(var) in range(min_, max_):
            Gaussian_thr = range_thr[range_t]
            break
        if var > max_:
            Gaussian_thr = 0
    if var < min_:
        Gaussian_thr = 1.8
    h, w = origin_img.size[1], origin_img.size[0]
    if max_v < tp_threshold:
        print(img_path, "template failed!!")
        if not im_flag:
            return
        else:
            max_pt = (w, h)
    im = origin_img.convert('L')
    start = ImageStat.Stat(im)
    bright_thr = int(start.mean[0])
    count = 0
    for bad_img in bad_imgs:
        roi2_img_path = os.path.join(bad_templates_path, bad_img)
        roi_im_bin = Image.open(roi2_img_path)
        for x in range(random.randint(10, 200), w - width, random.randint(80, 200)):
            for y in range(random.randint(10, 200), h - height, random.randint(80, 150)):
                if judge_in_area(h, w, x, y, max_pt) and judge_in_area(h, w, x + width, y, max_pt) and \
                        judge_in_area(h, w, x, y + height, max_pt) and judge_in_area(h, w, x + width, y + height,
                                                                                     max_pt):
                    new_pt = (x, y)
                    if count % 10 < train_thr * 10:
                        target_path = os.path.join(train_path, img_n[:-4] + "_" + str(count) + ".jpg")
                    elif count % 10 < (train_thr + val_thr) * 10:
                        target_path = os.path.join(val_path, img_n[:-4] + "_" + str(count) + ".jpg")
                    else:
                        target_path = os.path.join(test_path, img_n[:-4] + "_" + str(count) + ".jpg")
                    im_after = add_img_to_image(origin_img, roi_im_bin, bright_thr, new_pt, Gaussian_thr)
                    img_bef = origin_img.crop((new_pt[0], new_pt[1], new_pt[0] + width,
                                               new_pt[1] + height))
                    new_img = Image.new("RGB", (width * 2, height))
                    new_img.paste(im_after, (0, 0))
                    new_img.paste(img_bef, (width, 0))
                    # exit()
                    new_img.save(target_path)
                    count += 1


if __name__ == "__main__":
    im_flag = True  # true 为打复赛的水印
    ttf_path = "./huawenxinwei.ttf"
    ori_img_path = '../../Train_DataSet_final'  # 原始用于训练图片的路径
    choose_img_path = "../../Train_DataSet_final"  # 挑选出来的图片的路径
    roi_img_path = '/mnt/data/mwq_dir/mwq/rematch_watermask_template_bigger_1.jpg'  # 带水印的模板路径
    dst_path = "./rematch_jinzhifuyin_data_for_first_train"  # 目的路径
    bad_templates_path = "./bad_templates"
    bad_imgs = os.listdir(bad_templates_path)
    train_path = os.path.join(dst_path, "train")  # 用于训练的图片
    test_path = os.path.join(dst_path, "test")
    val_path = os.path.join(dst_path, "val")
    if not os.path.exists(os.path.join(dst_path, "train")):
        os.makedirs(os.path.join(dst_path, "train"))
    if not os.path.exists(os.path.join(dst_path, "test")):
        os.makedirs(os.path.join(dst_path, "test"))
    if not os.path.exists(os.path.join(dst_path, "val")):
        os.makedirs(os.path.join(dst_path, "val"))
    roi_im = Image.open(roi_img_path)
    width, height = roi_im.size[0], roi_im.size[1]
    tp_threshold = 0.5
    train_thr = 0.8
    val_thr = 0.1
    test_thr = 0.1
    ori_img_names = os.listdir(choose_img_path)
    for img_name in ori_img_names:
        gen_run(img_name)
    # n = 0
    # pool = Pool(processes=30)
    # pool.map(gen_run, ori_img_names)
    # pool.close()
    # pool.join()
    # print("%d images failed" % n)
