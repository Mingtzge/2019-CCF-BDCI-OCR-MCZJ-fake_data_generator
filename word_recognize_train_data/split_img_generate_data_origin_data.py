import os
import pandas as pd
from multiprocessing import Pool
import cv2

issuing_unit = {
    "x_d": 169,
    "y_d": 193,
    "w": 230,
    "h": 40,
    "index": 9
}
effective_data = {
    "x_d": 167,
    "y_d": 227,
    "w": 192,
    "h": 19,
    "index": 10
}
name = {
    "x_d": 85,
    "y_d": 39,
    "w": 106,
    "h": 24,
    "index": 1
}
gender = {
    "x_d": 85,
    "y_d": 72,
    "w": 24,
    "h": 24,
    "index": 3
}
nationality = {
    "x_d": 185,
    "y_d": 72,
    "w": 121,
    "h": 25,
    "index": 2
}
birthday_year = {
    "x_d": 84,
    "y_d": 105,
    "w": 47,
    "h": 21,
    "index": 4
}
birthday_month = {
    "x_d": 142,
    "y_d": 105,
    "w": 31,
    "h": 23,
    "index": 5
}
birthday_day = {
    "x_d": 198,
    "y_d": 105,
    "w": 29,
    "h": 22,
    "index": 6
}
address = {
    "x_d": 82,
    "y_d": 138,
    "w": 210,
    "h": 64,
    "index": 7
}
id_card = {
    "x_d": 131,
    "y_d": 221,
    "w": 246,
    "h": 24,
    "index": 8
}


def match_img(image, template, value):
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = value
    min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(res)
    if max_v < threshold:
        return False
    if max_pt[0] > 20 or max_pt[1] > 20:
        return False
    return max_pt


def crop_img(mark_point, args, ori_img, save_path, seq, label, type_c):
    try:
        x_p = mark_point[0] + args["x_d"]
        y_p = mark_point[1] + args["y_d"]
        c_img = ori_img[y_p:y_p + args["h"], x_p: x_p + args["w"]]
        if type_c == "Train":
            c_img_save_path = os.path.join(save_path, "%s_%s_%s.jpg" % (str(seq), label[args["index"]], str(args["index"])))
        elif type_c == "Test":
            c_img_save_path = os.path.join(save_path, "%s_%s_%s.jpg" % (str(seq), label, str(args["index"])))
        else:
            print("type invalid")
            return
        cv2.imwrite(c_img_save_path, c_img)
    except():
        return


def generate_data(ori_img_path, template, save_path, flag, thr_value, seq, label, type_c):
    ori_img = cv2.imread(ori_img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
    mark_point =[1,8]
    if flag == "0":
        crop_img(mark_point, issuing_unit, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, effective_data, ori_img, save_path, seq, label, type_c)
    else:
        crop_img(mark_point, name, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, gender, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, birthday_year, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, birthday_month, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, birthday_day, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, address, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, id_card, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, nationality, ori_img, save_path, seq, label, type_c)

def run_gen_train_data(label_path, final_save_path, template_base_path, origin_img_path):
    train_labels = pd.read_csv(label_path, header=None)
    template_img = cv2.imread(template_base_path, 0)
    img_names = os.listdir(origin_img_path)
    if not os.path.exists(final_save_path):
        os.makedirs(final_save_path)
    for count, img_name in enumerate(img_names):
        img_path = os.path.join(origin_img_path, img_name)
        names = img_name.split("_")
        label_info = None
        index = None
        for index, label_name in enumerate(train_labels[0:-1][0]):
            if label_name == names[0]:
                label_info = train_labels.loc[index]
                break
        if label_info is not None:
            #pool.apply_async(generate_data,(img_path, template_img, final_save_path, names[1][0], 0.4, index, label_info, "Train", ))
            generate_data(img_path, template_img, final_save_path, names[1][0], 0.4, index, label_info, "Train")

        else:
            print("%s     not find" % names[0])


def run_gen_test_data(final_save_path, template_base_path, origin_img_path):
    template_img = cv2.imread(template_base_path, 0)
    img_names = os.listdir(origin_img_path)
    if not os.path.exists(final_save_path):
        os.makedirs(final_save_path)
    pool = Pool(processes=30)
    for count, img_name in enumerate(img_names):
        img_path = os.path.join(origin_img_path, img_name)
        names = img_name.split("_")
        pool.apply_async(generate_data,
                             (img_path, template_img, final_save_path, names[1][0], 0.2, count, img_name[:-4], "Test", ))
    pool.close()
    pool.join()

