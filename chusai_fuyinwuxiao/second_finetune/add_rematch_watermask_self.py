#! encoding:utf-8
import torchvision.transforms as transforms
from multiprocessing import Pool
import os
import numpy as np
import random
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageStat, ImageEnhance, ImageFilter  # 导入模块
from PIL import ImageFile

# import


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


def add_text_to_image(image, text, p):
    font = ImageFont.truetype(ttf_path, 42)
    # 添加背景
    new_img = Image.new('RGBA', (image.size[0], image.size[1]), (0, 0, 0, 0))
    new_img.paste(image)
    # new_img.show()

    # 添加水印
    font_len = len(text)
    rgba_image = new_img.convert('RGBA')
    text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
    image_draw = ImageDraw.Draw(text_overlay)

    b_thr = random.randint(110, 190)
    bright_random = min(random.randint(b_thr + 10, b_thr + 25), 170)
    # print(bright_random)
    x = max(int(np.random.normal(12, 8)), 0)
    y = max(int(np.random.normal(11, 7)), 0)
    if x > width - 178:
        x = width - 178
    if y > height - 51:
        y = height - 51
    p_new = (p[0] + x, p[1] + y)

    pix_rand = random.randint(10, 120)
    image_draw.rectangle((p_new, (p_new[0] + 177, p_new[1] + 50)),
                         outline=(pix_rand, pix_rand, pix_rand, bright_random), width=4)
    image_draw.text((p_new[0] + 5, p_new[1] + 5), "复印无效", font=font,
                    fill=(pix_rand, pix_rand, pix_rand, bright_random), )  # 利用ImageDraw的内置函数，在图片上写入文字

    text_overlay_postion = []  # 水印位置
    sound_flag = True
    flag_rand = random.randint(0, 3)
    if flag_rand == 0:
        conF = text_overlay.filter(ImageFilter.CONTOUR)  ##找轮廓
        text_overlay_array = np.array(conF)

    if flag_rand == 1:
        conF = text_overlay.filter(ImageFilter.CONTOUR)  ##找轮廓
        text_overlay_array = np.array(conF)

    if flag_rand == 2:
        text_overlay_array = np.array(text_overlay)

    if flag_rand == 3:
        sound_flag = False

    if sound_flag:
        for i in range(0, text_overlay.size[0]):
            for j in range(0, text_overlay.size[1]):
                if text_overlay_array[j][i][0] < 240:
                    text_overlay_postion.append((i, j))

        for i in range(0, len(text_overlay_postion)):
            overlay_x = text_overlay_postion[i][0]
            overlay_y = text_overlay_postion[i][1]
            new_bri = random.randint(bright_random - 20, bright_random + 50)
            color = text_overlay.getpixel((overlay_x, overlay_y))
            color = color[:-1] + (new_bri,)
            text_overlay.putpixel((overlay_x, overlay_y), color)

    img_bef = rgba_image.crop((p[0], p[1], p[0] + width, p[1] + height))
    enh_sha = ImageEnhance.Sharpness(text_overlay)
    sharpness = 1
    text_overlay = enh_sha.enhance(sharpness)
    text_overlay = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0)(text_overlay)
    text_overlay = text_overlay.filter(MyGaussianBlur(radius=0.1))

    image_with_text = Image.alpha_composite(rgba_image, text_overlay)
    img_after = image_with_text.crop((p[0], p[1], p[0] + width, p[1] + height))
    new_img = Image.new("RGB", (width * 2, height))
    new_img.paste(img_after, (0, 0))
    new_img.paste(img_bef, (width, 0))
    # 裁切图片
    # image_with_text = image_with_text.crop(
    #     (image.size[0] + p[0], image.size[1] + p[1], image.size[0] + p[0] + width, image.size[1] + p[1] + height))
    return new_img

    ###
    # 伪造去水印训练数据集
    ###


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


def getImageVar(imgPath):
    image = cv2.imread(imgPath)
    cv2.imshow(imgPath, image)
    cv2.waitKey(0)
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


def run(filename):
    for i in range(15):
        pt_x = random.randint(0, 220)
        pt_y = random.randint(0, 220)
        new_pt = (pt_x, pt_y)
        image = Image.open(path + '/' + filename)

        img2gray = image.convert('L')
        start = ImageStat.Stat(img2gray)
        bright_thr = int(start.mean[0])

        img2gray = add_text_to_image(img2gray, u'复印无效', new_pt)

        bri_rand = random.randint(129, 198)  # 随机亮度
        enh_bri = ImageEnhance.Brightness(img2gray)
        brightness = bri_rand / bright_thr
        img2gray = enh_bri.enhance(brightness)

        sharp_rand = random.uniform(1, 4)  # 随机锐化
        enh_sha = ImageEnhance.Sharpness(img2gray)
        sharpness = sharp_rand
        img2gray = enh_sha.enhance(sharpness)

        filter_rand = random.uniform(0.2, 1.3)  # 随机高斯模糊
        img2gray = img2gray.filter(MyGaussianBlur(radius=filter_rand))

        img_res = cv2.cvtColor(np.asarray(img2gray), cv2.COLOR_RGB2BGR)
        imgVar = cv2.Laplacian(img_res, cv2.CV_64F).var()
        if i % 10 < 8:
            cv2.imwrite(os.path.join(dst_path, "train", r'{}_{}.jpg'.format(filename[:-4], i)), img_res)
        elif i % 10 < 9:
            cv2.imwrite(os.path.join(dst_path, "val", r'{}_{}.jpg'.format(filename[:-4], i)), img_res)
        elif i % 10 < 10:
            cv2.imwrite(os.path.join(dst_path, "test", r'{}_{}.jpg'.format(filename[:-4], i)), img_res)
    print(filename)


if __name__ == "__main__":
    cnt = 0
    path = './fake_real_color_data'
    dst_path = "./fake_data_for_finetune"
    sub_dirs = ["test", "train", "val"]
    for sub_dir in sub_dirs:
        if not os.path.exists(os.path.join(dst_path, sub_dir)):
            os.makedirs(os.path.join(dst_path, sub_dir))
    filenames = os.listdir(path)
    ttf_path = "STXINWEI.TTF"
    roi_img_path = 'chusai_watermask_template.jpg'  # 带水印的模板路径
    roi_im = Image.open(roi_img_path)
    width, height = roi_im.size[0], roi_im.size[1]
    for filename in filenames:
        run(filename)