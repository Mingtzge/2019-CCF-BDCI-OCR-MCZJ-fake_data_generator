#! encoding:utf-8
import torchvision.transforms as transforms
import os
import random
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageStat, ImageEnhance, ImageFilter  # 导入模块


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


def add_text_to_image(image, text, b_thr, p):
    font = ImageFont.truetype(ttf_path, 42)
    # 添加背景
    new_img = Image.new('RGBA', (image.size[0] * 2, image.size[1] * 2), (0, 0, 0, 0))
    new_img.paste(image, image.size)
    # new_img.show()

    # 添加水印
    font_len = len(text)
    rgba_image = new_img.convert('RGBA')
    text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
    image_draw = ImageDraw.Draw(text_overlay)
    bright_random = min(random.randint(b_thr + 10, b_thr + 25), 190)
    p_new = (p[0] + image.size[0] + 12, p[1] + image.size[1] + 11)
    #         image_draw.text((i, j), text, font=font, fill=(0, 0, 0, 80))
    image_draw.rectangle((p_new, (p_new[0] + 177, p_new[1] + 50)), outline=(0, 0, 0, bright_random), width=4)
    image_draw.text((p_new[0] + 5, p_new[1] + 5), "复印无效", font=font,
                    fill=(0, 0, 0, bright_random))  # 利用ImageDraw的内置函数，在图片上写入文字
    # text_overlay = text_overlay.rotate(-45)
    enh_sha = ImageEnhance.Sharpness(text_overlay)
    sharpness = 1
    text_overlay = enh_sha.enhance(sharpness)
    text_overlay = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0)(text_overlay)
    text_overlay = text_overlay.filter(MyGaussianBlur(radius=0.1))
    # text_overlay.show()
    image_with_text = Image.alpha_composite(rgba_image, text_overlay)

    # 裁切图片
    image_with_text = image_with_text.crop(
        (image.size[0] + p[0], image.size[1] + p[1], image.size[0] + p[0] + width, image.size[1] + p[1] + height))
    return image_with_text

    ###
    # 伪造去水印训练数据集
    ###


def judge_in_area(h, w, x, y, pt):
    if ((x in range(pt[0], pt[0] + width)) and (y in range(pt[1], pt[1] + height))) or x > w or y > h:
        return False
    return True


def gen_run(img_n):
    print(img_n)
    img_path = os.path.join(choose_img_path, img_n)
    img_rgb = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(roi_img_path, 0)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(res)
    origin_img = Image.open(img_path)
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
    for x in range(random.randint(10, 200), w - width, random.randint(80, 200)):
        for y in range(random.randint(10, 200), h - height, random.randint(80, 150)):
            if judge_in_area(h, w, x, y, max_pt) and judge_in_area(h, w, x + width, y, max_pt) and \
                    judge_in_area(h, w, x, y + height, max_pt) and judge_in_area(h, w, x + width, y + height, max_pt):
                new_pt = (x, y)
                if count % 10 < train_thr * 10:
                    target_path = os.path.join(train_path, img_n[:-4] + "_" + str(count) + ".jpg")
                elif count % 10 < (train_thr + val_thr) * 10:
                    target_path = os.path.join(val_path, img_n[:-4] + "_" + str(count) + ".jpg")
                else:
                    target_path = os.path.join(test_path, img_n[:-4] + "_" + str(count) + ".jpg")
                im_after = add_text_to_image(origin_img, u'复印无效', bright_thr, new_pt)
                img_bef = origin_img.crop((new_pt[0], new_pt[1], new_pt[0] + width,
                                       new_pt[1] + height))
                new_img = Image.new("RGB", (width * 2, height))
                new_img.paste(im_after, (0, 0))
                new_img.paste(img_bef, (width, 0))
                new_img.save(target_path)
                count += 1


if __name__ == "__main__":
    im_flag = False  # "复印无效水印"
    ttf_path = "./huawenxinwei.ttf"
    ori_img_path = '../../Train_DataSet_final'  # 复赛和初赛所有的训练数据
    choose_img_path = ori_img_path
    roi_img_path = './chusai_watermask_template_bigger_1.jpg'  # 带水印的模板路径
    dst_path = "./fake_data_for_first_train/"  # 目的路径
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
    test_thr = 0.1 # 各个数据集的比例
    ori_img_names = os.listdir(choose_img_path)
    for img in ori_img_names:
        gen_run(img)
