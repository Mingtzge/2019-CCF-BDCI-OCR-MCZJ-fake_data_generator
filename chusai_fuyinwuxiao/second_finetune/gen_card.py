import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv
import os


# 伪造正面
def gen_card_front(img, str):
    font1 = os.path.join(ori_path, r'src\black.TTF')
    font2 = os.path.join(ori_path, r'src\msyh.ttc')
    color_blue = (0, 191, 255)
    color_black = (0, 0, 0)

    img_res = img_put_text(img, '姓  名', 38, 53, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[0], 92, 49, font1, color_black, 19)
    img_res = img_put_text(img_res, '性  别', 38, 86, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[1], 90, 81, font1, color_black, 19)
    img_res = img_put_text(img_res, '民  族', 136, 86, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[2], 191, 81, font1, color_black, 19)
    img_res = img_put_text(img_res, '出  生', 38, 118, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[3], 89, 114, font1, color_black, 19)
    img_res = img_put_text(img_res, '年', 136, 118, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[4], 154, 114, font1, color_black, 19)
    img_res = img_put_text(img_res, '月', 181, 118, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[5], 204, 114, font1, color_black, 19)
    img_res = img_put_text(img_res, '日', 226, 118, font1, color_blue, 13)
    img_res = img_put_text(img_res, '住  址', 38, 150, font1, color_blue, 13)

    shenfen_list = ['公', '民', '身', '份', '号', '码']
    for i in range(0, len(shenfen_list)):
        img_res = img_put_text(img_res, shenfen_list[i], 38 + i * 14, 235, font1, color_blue, 13)

    if len(str[6]) > 12:
        addr_list1 = list(str[6][0:12])
        for i in range(0, len(addr_list1)):
            img_res = img_put_text(img_res, addr_list1[i], 87 + i * 17, 146, font1, color_black, 19)

        if len(str[6]) > 24:
            addr_list2 = list(str[6][12:24])
            for i in range(0, len(addr_list2)):
                img_res = img_put_text(img_res, addr_list2[i], 87 + i * 17, 168, font1, color_black, 19)

            addr_list3 = list(str[6][24:len(str[6])])
            for i in range(0, len(addr_list3)):
                img_res = img_put_text(img_res, addr_list3[i], 87 + i * 17, 190, font1, color_black, 19)

        else:
            addr_list2 = list(str[6][12:len(str[6])])
            for i in range(0, len(addr_list2)):
                img_res = img_put_text(img_res, addr_list2[i], 87 + i * 17, 168, font1, color_black, 19)

    else:
        addr_list = list(str[6])
        for i in range(0, len(addr_list)):
            img_res = img_put_text(img_res, addr_list[i], 87 + i * 17, 146, font1, color_black, 19)

    id_list = list(str[7])
    for i in range(0, len(id_list)):
        img_res = img_put_text(img_res, id_list[i], 136 + i * 13.4, 227, font2, color_black, 19)

    return img_res


# 伪造背面
def gen_card_back(img, str):
    font1 = os.path.join(ori_path, r'src\black.TTF')

    color_black = (0, 0, 0)
    img_res = img_put_text(img, '签发机关', 108, 202, font1, color_black, 15)

    if len(str[0]) > 12:
        institu_list = list(str[0][0:12])
        for i in range(0, len(institu_list)):
            img_res = img_put_text(img_res, institu_list[i][0:12], 175 + i * 17, 200, font1, color_black, 19)

        institu_list2 = list(str[0][12:len(str[0])])
        for i in range(0, len(institu_list2)):
            img_res = img_put_text(img_res, institu_list2[i][0:12], 175 + i * 17, 219, font1, color_black, 19)

    else:
        institu_list = list(str[0])
        for i in range(0, len(institu_list)):
            img_res = img_put_text(img_res, institu_list[i], 175 + i * 17, 200, font1, color_black, 19)

    img_res = img_put_text(img_res, '有效期限', 108, 236, font1, color_black, 15)

    date_list1 = list(str[1][0:4])
    for i in range(0, len(date_list1)):
        img_res = img_put_text(img_res, date_list1[i], 176 + i * 10, 233, font1, color_black, 19)

    img_res = img_put_text(img_res, '.', 215, 233, font1, color_black, 19)

    date_list2 = list(str[1][5:7])
    for i in range(0, len(date_list2)):
        img_res = img_put_text(img_res, date_list2[i], 219 + i * 10, 233, font1, color_black, 19)

    img_res = img_put_text(img_res, '.', 239, 233, font1, color_black, 19)

    date_list3 = list(str[1][8:10])
    for i in range(0, len(date_list3)):
        img_res = img_put_text(img_res, date_list3[i], 243 + i * 10, 233, font1, color_black, 19)

    img_res = img_put_text(img_res, '-', 263, 235, font1, color_black, 15)

    # 如果最后是数字
    str_temp = str[1][11:len(str[1])]
    if str_temp != '长期':

        str_temp_list1 = list(str_temp[0:4])
        for i in range(0, len(str_temp_list1)):
            img_res = img_put_text(img_res, str_temp_list1[i], 270 + i * 10, 233, font1, color_black, 19)

        img_res = img_put_text(img_res, '.', 309, 233, font1, color_black, 19)

        str_temp_list2 = list(str_temp[5:7])
        for i in range(0, len(str_temp_list2)):
            img_res = img_put_text(img_res, str_temp_list2[i], 313 + i * 10, 233, font1, color_black, 19)

        img_res = img_put_text(img_res, '.', 333, 233, font1, color_black, 19)

        str_temp_list3 = list(str_temp[8:10])
        for i in range(0, len(str_temp_list3)):
            img_res = img_put_text(img_res, str_temp_list3[i], 337 + i * 10, 233, font1, color_black, 19)

    else:
        img_res = img_put_text(img_res, '长期', 271.5, 233, font1, color_black, 18)

    return img_res


# 生成画布
def img_to_white(img):
    h = img.shape[0]
    w = img.shape[1]
    target = np.ones((h, w), dtype=np.uint8) * 255
    ret = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    for i in range(h):
        for j in range(w):
            ret[i, j, 0] = img[i, j, 0]
            ret[i, j, 1] = img[i, j, 1]
            ret[i, j, 2] = img[i, j, 2]
    return ret


# 在画布上书写文字
def img_put_text(img, str, pos_x, pos_y, font, color, size):
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mfont = ImageFont.truetype(font, size)
    fillColor = color
    position = (pos_x, pos_y)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, str, font=mfont, fill=fillColor)
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img_OpenCV


def gen_faker_card_run():
    csv_file = open(ori_csv_file, 'r', encoding='UTF-8')
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    date = []  # 创建列表准备接收csv各行数据
    cnt = 0  # 记录csv文件行数

    for one_line in csv_reader_lines:
        date.append(one_line)

        result_front = []
        result_back = []

        result_front.append(date[cnt][1])  # 姓名
        result_front.append(date[cnt][3])  # 性别
        result_front.append(date[cnt][2])  # 名族
        result_front.append(date[cnt][4])  # 年
        result_front.append(date[cnt][5])  # 月
        result_front.append(date[cnt][6])  # 日
        result_front.append(date[cnt][7])  # 地址
        result_front.append(date[cnt][8])  # 身份号

        result_back.append(date[cnt][9])  # 签发机关
        result_back.append(date[cnt][10])  # 有效日期

        image1 = cv2.imread(front_img)  # 读取正面模板
        image2 = cv2.imread(back_img)  # 读取背面模板

        img_new_white1 = img_to_white(image1)  # 生成画布
        img_res_f = gen_card_front(img_new_white1, result_front)  # 写入文字
        cv2.imwrite(result_card_path + '\{}_1.jpg'.format(cnt), img_res_f)

        img_new_white2 = img_to_white(image2)
        img_res_b = gen_card_back(img_new_white2, result_back)
        cv2.imwrite(result_card_path + '\{}_0.jpg'.format(cnt), img_res_b)
        cnt = cnt + 1
        print(cnt)


if __name__ == "__main__":
    ori_path = os.path.abspath('.')
    front_img = os.path.join(ori_path, r'src\front.png')  # 正面模板
    back_img = os.path.join(ori_path, r'src\back.png')  # 背面模板
    result_card_path = os.path.join(ori_path, r'fake_real_color_data')  # 伪造原始身份证样本结果路径
    csv_files = ["Train_Labels.csv", "Train_Labels2.csv"]
    for csv_file in csv_files:
        ori_csv_file = os.path.join(ori_path, r'src', csv_file)  # 原始数据
        gen_faker_card_run()
