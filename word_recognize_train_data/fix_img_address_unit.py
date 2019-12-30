import numpy as np
import cv2
import sys
import os


def preprocess_img(img, name):
    resize_img = cv2.resize(img, (int(2.0 * img.shape[1]), int(2.0 * img.shape[0])), interpolation=cv2.INTER_CUBIC)
    # 放大两倍，更容易识别
    resize_img = cv2.convertScaleAbs(resize_img, alpha=0.35, beta=20)
    resize_img = cv2.normalize(resize_img, dst=None, alpha=300, beta=10, norm_type=cv2.NORM_MINMAX)
    img_blurred = cv2.medianBlur(resize_img, 7)  # 中值滤波
    img_blurred = cv2.medianBlur(img_blurred, 3)
    # 这里面的几个参数，alpha，beta都可以调节，目前感觉效果还行，但是应该还可以调整地更好

    return img_blurred


def detect_fn(img, img_name, img_save_path):
    resize_img = cv2.resize(img, (int(2.0 * img.shape[1]), int(2.0 * img.shape[0])), interpolation=cv2.INTER_CUBIC)
    img = preprocess_img(img, img_name)
    # cv2.imwrite(img_save_path + img_name + '_processed.jpg', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 6))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 4))  # 两个参数可调
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)
    dilation2 = cv2.dilate(erosion, element2, iterations=2)

    # cv2.imwrite(img_save_path + img_name + '_dilation.jpg', dilation2)

    region = []
    #  查找轮廓
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 利用以上函数可以得到多个轮廓区域，存在一个列表中。
    #  筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area < 50):
            continue
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 筛选那些太细的矩形，留下扁的
        if 25 < height < 80 and width > 25 and height < width * 1.3:
            region.append(box)
    max_x = 0
    for box in region:  # 每个box是左下，左上，右上，右下坐标
        for box_p in box:
            if box_p[0] > max_x:
                max_x = box_p[0]
    h, w, c = resize_img.shape
    # cv2.rectangle(resize_img, (0,0), (max_x, h),  (0, 0, 255))
    # cv2.imwrite(img_save_path + img_name + '_res.jpg', resize_img)
    # region = process_region(region)
    return resize_img[0:h, 0:min(max_x + 50, w)]


def merge_img(img_path, points):
    img = cv2.imread(img_path)
    img_count = len(points)
    image3 = np.hstack([img[points[0][0][1]:points[0][1][1], points[0][0][0]:points[0][1][0]],
                        img[points[1][0][1]:points[1][1][1], points[1][0][0]:points[1][1][0]]])
    if img_count == 3:
        image3 = np.hstack([image3, img[points[2][0][1]:points[2][1][1], points[2][0][0]:points[2][1][0]]])
    return image3


def fix_address_unit(train_data_path, save_data_path):
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    if not os.path.exists(train_data_path):
        print("not  exist train data ,exit...")
        return
    train_img_names = os.listdir(train_data_path)
    type_7 = [[(0, 0), (209, 21)], [(5, 21), (209, 42)], [(5, 42), (209, 63)]]
    type_9 = [[(0, 0), (229, 20)], [(5, 19), (229, 39)]]
    for index, train_img_name in enumerate(train_img_names):
        if train_img_name[-5] in ["9", "7"]:
            os.system("cp %s %s" % (os.path.join(train_data_path, train_img_name), os.path.join(save_data_path, train_img_name)))
            p = type_9
            if train_img_name[-5] == "7":
                p = type_7
            try:
                new_img = merge_img(os.path.join(train_data_path, train_img_name), p)
                img = detect_fn(new_img, train_img_name, train_data_path)
                # cv2.imshow(train_img_name,new_img)
                cv2.imwrite(os.path.join(train_data_path, train_img_name), img)
            except Exception:
                print("修复图片出错：--->", train_img_name)


#if __name__ == "__main__":
#    ori_data_path = "/home/jiangmingzhi/mnt/mnt/data/mwq_dir/CCF_CBDI_TestData/gen_test_data_10_6/train_data_dir"
#    bak_data_path = "/home/jiangmingzhi/mnt/mnt/data/mwq_dir/CCF_CBDI_TestData/gen_test_data_10_6/check_data"
#    fix_address_unit(ori_data_path, bak_data_path)

