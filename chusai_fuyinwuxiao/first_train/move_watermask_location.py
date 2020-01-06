import cv2
import numpy as np
import copy
import os
import random


def merge_img(image1, image2):
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    if h1 != h2 or w1 != w2:
        image2 = cv2.resize(image2, (w1, h1))
    image3 = np.hstack([image2, image1])
    return image3


# 得到水印所在位置和输入图片灰度图
def match_img(image, target, value):
    img_rgb = cv2.imread(image)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(target, 0)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = value
    min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(res)
    if max_v < threshold:
        return False, False
    return max_pt, img_gray


####
# new_pt:新的子图起始坐标点
# origin_image:原始图片
# target_path:目的存放路径
# img_father:含有原始水印的图,用于从上面提取像素值
# father_cor:原始水印坐标起始点
####
def water_add(new_pt, origin_image, target_path, img_father, father_cor):
    no_water_image = copy.deepcopy(origin_image[new_pt[1]:new_pt[1] + height, new_pt[0]:new_pt[0] + width])
    water_image = copy.deepcopy(no_water_image)
    x = int(np.random.normal(0, 8))
    y = int(np.random.normal(0, 8))
    if x < -min(x_p):
        x = -min(x_p)
    if x > width - max(x_p) - 1:
        x = width - max(x_p) - 1
    if y < -min(y_p):
        y = -min(y_p)
    if y > height - max(y_p) - 1:
        y = height - max(y_p) - 1
    h, w = water_image.shape
    for p in water_points:
        p_value = img_father[father_cor[1] + p[0], father_cor[0] + p[1]]
        if water_image[p[0] + y, p[1] + x] > p_value:
            water_image[p[0] + y, p[1] + x] = p_value  # 如果打水印的地方的像素值高于水印的像素值,则极可能没有文字重叠,直接用水印的像素值覆盖,否则,不覆盖,也就是说,谁最黑,用谁
        # elif p_value < 125: # 原本打算取比例结合,可能是没有找到合适的方式,效果并没有提高
        #    water_image[p[0], p[1]] = (water_image[p[0], p[1]] + p_value)*0.4
    new_image = merge_img(no_water_image, water_image)  # 将无水印的图和有水印的水平拼接,训练时要求的格式
    cv2.imwrite(target_path, new_image)


# 判断子图是与原始水印位置重叠
def judge_in_area(h, w, x, y, pt):
    if ((x in range(pt[0], pt[0] + width)) and (y in range(pt[1], pt[1] + height))) or x + width > w or y + height > h:
        return False
    return True


####
# origin_img_path:
# image_name: 图片信息,用于获得正反面两张图片
# img_father: 原始水印所在的图片
# father_cor: 原始水印坐标点
####
def gen_data(origin_img_path, image_name, img_father, father_cor):
    water_pt, origin_image_gray = match_img(origin_img_path, roi_img_path, tp_threshold)
    if water_pt is False:
        print(image_name)
        return
    h, w = origin_image_gray.shape
    count = 0
    for x in range(10, w - width, random.randint(5, 10)):  # 移动子图片起始坐标点,获取子图
        for y in range(20, h - height, random.randint(5, 10)):
            if judge_in_area(h, w, x, y, water_pt) and judge_in_area(h, w, x + width, y, water_pt) and \
                    judge_in_area(h, w, x, y + height, water_pt) and judge_in_area(h, w, x + width, y + height,
                                                                                   water_pt):
                new_pt = [x, y]
                if count % 10 < train_thr * 10:
                    target_path = os.path.join(train_path, image_name[:-4] + "_" + str(count) + "_ori.jpg")
                elif count % 10 < (train_thr + val_thr) * 10:
                    target_path = os.path.join(val_path, image_name[:-4] + "_" + str(count) + "_ori.jpg")
                else:
                    target_path = os.path.join(test_path, image_name[:-4] + "_" + str(count) + "_ori.jpg")
                water_add(new_pt, origin_image_gray, target_path, img_father, father_cor)  # 给图片打水印
                count += 1


def m_run(name):
    print(name)
    img_path = os.path.join(choose_img_path, name)
    father_cor, img_father = match_img(img_path, roi_img_path, tp_threshold)
    if father_cor is False:
        print(name)
        return
    gen_data(img_path, name, img_father, father_cor)  # 打水印生成伪造子图片
    if name[-5] == "0":
        name = name[:-5] + "1" + name[-4:]
    else:
        name = name[:-5] + "0" + name[-4:]
    img_path = os.path.join(ori_img_path, name)
    if os.path.exists(img_path): 
        gen_data(img_path, name, img_father, father_cor)


###
# 伪造去水印训练数据集
###
if __name__ == "__main__":
    ori_img_path = '../../Train_DataSet_final/'  # 原始用于训练图片的路径
    choose_img_path = "./choosed_imgs_10_2/"  # 挑选出来的图片的路径
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
    roi_img = cv2.imread(roi_img_path)
    roi_img_gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    ret, roi_img_bin = cv2.threshold(roi_img_gray, 175, 255, cv2.THRESH_BINARY)  # 二值化
    water_points = []
    width, height = roi_img_bin.shape[::-1]
    for w in range(width):
        for h in range(height):
            if roi_img_bin[h][w] == 0:
                water_points.append([h, w])  # 记录水印的位置和对应的像素值
    x_p = [p_[1] for p_ in water_points]
    y_p = [p_[0] for p_ in water_points]
    tp_threshold = 0.1
    train_thr = 0.8
    val_thr = 0.1
    test_thr = 0.
    ori_img_names = os.listdir(choose_img_path)
    n = 0
    for img in ori_img_names:
        m_run(img)
    print("%d images failed" % n)
