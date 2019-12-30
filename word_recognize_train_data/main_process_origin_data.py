import cv2
import numpy as np
from multiprocessing import Pool
#from split_img_generate_data import run_gen_train_data, run_gen_test_data
from .split_img_generate_data_origin_data import run_gen_train_data, run_gen_test_data
from .fix_img_address_unit import fix_address_unit
from .preprocess_for_test import preprocess_imgs
import copy
import os


def merge_img(image1, image2):
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    if h1 != h2 or w1 != w2:
        image2 = cv2.resize(image2, (w1, h1))
    image3 = np.hstack([image2, image1])
    return image3


def match_img(image, target, value, rematch = False):
    img_rgb = cv2.imread(image)
    h, w, c = img_rgb.shape
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(target, 0)
    th,tw = template.shape
    max_v1 = 0
    if not rematch:
        template = template[16:56, 20:186]
    else:
        template = template[18:107, 19:106]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = value
    min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(res)
    if max_v < threshold:
        return False, False, False
    if not rematch:
        template1 = cv2.imread(roi_rematch_img_path, 0)
        template1 = template1[18:107, 19:106]
        res1 = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_v1, max_v1, min_pt1, max_pt1 = cv2.minMaxLoc(res1)
        if max_v < max_v1: # 避免两种水印匹配重叠的情况
        	return False, False, False
    if not rematch:
        x = 20
        y = 16
    else:
        x = 19
        y = 18
    ori_pt = (min(w-tw-1,max(max_pt[0] - x, 0)), max(0,min(max_pt[1] - y, h - th - 1)))
    return ori_pt, img_gray, max_pt


def judge_in_area(x, y, pt):
    if (x in range(pt[0], pt[0] + width)) and (y in range(pt[1], pt[1] + height)):
        return False
    return True


def gen_align_real_test_data(rematch = False):
    origin_img_names = os.listdir(ori_img_path)
    if pool_num > 0:
        pool = Pool(pool_num)
        if args.debug:
            pool.map(gen_test_data_run, origin_img_names[:pool_num])
        else:
            pool.map(gen_test_data_run, origin_img_names)
        pool.close()
        pool.join()
    else:
        for origin_img in origin_img_names:
            gen_test_data_run(origin_img)

def gen_test_data_run(ori_name):
    idx = 0
    #for idx, ori_name in enumerate(origin_img_names):
    if True:
        img_path = os.path.join(ori_img_path, ori_name)
        pt, img_gray, m_p = match_img(img_path, roi_img_path, 0.3, rematch)
        if pt is False:
            #print(img_path)
            #continue
            return
        new_img = merge_img(img_gray[pt[1]:pt[1] + height, pt[0]:pt[0] + width],
                            img_gray[pt[1]:pt[1] + height, pt[0]:pt[0] + width])
        if not rematch:
            flag = "_chu_" + str(pt[0]) + "_" +  str(pt[1]) + "_"  +  str(m_p[0]) + "_" +  str(m_p[1]) + "_chu_"
        else:
            flag = "_fu_" + str(pt[0]) + "_" +  str(pt[1]) + "_"  + str(m_p[0]) + "_" +  str(m_p[1]) + "_fu_"
        cv2.imwrite(os.path.join(test_data_dst_path, "test", ori_name[:-4] + flag + ".jpg"), new_img)


def gan_gen_result(rematch = False):
    if not rematch:
        size_ = size_chu
    else:
        size_ = size_fu
    test_img_count = len(os.listdir(os.path.join(test_data_dst_path, "test")))
    if test_img_count > 0 and os.path.exists(test_data_dst_path):
        os.system(
            "python3.6 ~/miaowq/jmz_workspace/pytorch-CycleGAN-and-pix2pix/test.py --dataroot %s --name %s\
             --model pix2pix --direction AtoB --checkpoints_dir %s\
              --num_test %s --results_dir %s --load_size %s --crop_size %s" % (test_data_dst_path, pixel_mode, args.checkpoints_dir, test_img_count, gan_result_dir, size_[0], size_[1]))
    else:
        print("there are something wrong in test_data_dst_path, exit...")
        exit(0)


def recover_origin_img(rematch = False):
    result_dir = os.path.join(gan_result_dir, pixel_mode, "test_latest", "images")
    if not os.path.exists(result_dir):
        print("not exists gan result dir, exit...")
        exit(0)
    result_img_names = os.listdir(result_dir)
    recovered_imgs = []
    for result_img_name in result_img_names:
        if "fake" in result_img_name:
            target_img_name = result_img_name.split("_")[0] + "_" + result_img_name.split("_")[1] + ".jpg"
            if rematch:
                target_img_name = result_img_name.split("_fu_")[0] + ".jpg"
            #pt, target_img, m_p = match_img(os.path.join(ori_img_path, target_img_name), roi_img_path, 0.1)
            #if pt is False:
            #    print(os.path.join(ori_img_path, target_img_name), "failed")
            #    return
            if not rematch:
                if not "_chu_" in result_img_name:
                    print("img name invalid!!-----------------------------<>----------------------------", result_img_name)
                    continue
                pts = result_img_name.split("_chu_")[1].split("_")
            else:
                if not "_fu_" in result_img_name:
                    print("img name invalid!!-----------------------------<>----------------------------", result_img_name)
                    continue
                pts = result_img_name.split("_fu_")[1].split("_")
            pt = (int(pts[0]), int(pts[1]))
            result_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(result_dir, result_img_name)), cv2.COLOR_BGR2GRAY), ( width, height))
            target_img = cv2.cvtColor(cv2.imread(os.path.join(ori_img_path, target_img_name)), cv2.COLOR_BGR2GRAY)
            #target_img[pt[1]:height, pt[0]:width] = result_img #图片替换
            try:
                for i in range(height):
                    for j in range(width):
                        target_img[pt[1] + i, pt[0] + j] = result_img[i, j] 
            #for p in water_points:
            #    target_img[p[0] + pt[1], p[1] + pt[0]] = result_img[p[0], p[1]]

                cv2.imwrite(os.path.join(recover_image_dir, result_img_name[:-11] + ".jpg"), target_img)
                recovered_imgs.append(target_img_name)
            except Exception:
                print("恢复图片异常：",result_img_name)
    ori_imgs = os.listdir(ori_img_path)
    print("原始图片数量:", len(ori_imgs), "恢复图片数量:", len(recovered_imgs))
    if len(ori_imgs) > len(recovered_imgs) and not args.debug:
        for img in ori_imgs:
            if img not in recovered_imgs:
                os.system("cp %s %s"%(os.path.join(ori_img_path, img), recover_image_dir + "/"))

if __name__ == "__main__":
    import argparse
    rematch = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_gen_data_chu", action="store_true", help="generate chusai new test data")
    parser.add_argument("--no_gen_data_fu", action="store_true", help="generate fusai new test data")
    parser.add_argument("--no_preprocessed", action="store_true", help="if preprocessed test data")
    parser.add_argument("--no_gan_test", action="store_true", help="test data with gan model")
    parser.add_argument("--no_gan_test_rematch", action="store_true", help="test rematch data with gan model")
    parser.add_argument("--no_rec_img", action="store_true", help="if recover img")
    parser.add_argument("--no_rec_img_rematch", action="store_true", help="if recover img")
    parser.add_argument("--no_train_data", action="store_true", help="if generate train data")
    parser.add_argument("--no_test_data", action="store_true", help="if generate test data")
    parser.add_argument("--no_fix_img", action="store_true", help="if fix img of address and unit")
    parser.add_argument("--debug", action="store_true", help="if recover img")
    parser.add_argument("--gan_chu", default="chusai_watermask_remover_model", help="model name of chusai")
    parser.add_argument("--gan_fu", default="fusai_watermask_remover_model", help="model name of fusai")
    parser.add_argument("--checkpoints_dir", default="../../pytorch-CycleGAN-and-pix2pix/models_data",
                        help="experience name of training")
    parser.add_argument("--pool_num", default=0, help="the number of threads for process data")
    args = parser.parse_args()
    size_chu = (512, 512)
    size_fu = (256, 256)
    pixel_mode = args.gan_chu
    pool_num = int(args.pool_num)
    header_dir = "large_data_for_crnn"
    dirs = ["chusai_data_for_watermark_remove/test", "fuusai_data_for_watermark_remove/test", "gan_result_chu_dir",
            "gan_result_fu_dir", "recover_image_chu_dir", "recover_image_fu_dir", "train_data_dir",
            "test_data_preprocessed", "test_data_txts"]
    test_data_dst_path = os.path.join(header_dir, "chusai_data_for_watermark_remove")
    gan_result_dir = os.path.join(header_dir, "gan_result_chu_dir")
    recover_image_dir = os.path.join(header_dir, "recover_image_chu_dir")
    train_data_dir = os.path.join(header_dir, "train_data_dir")
    fix_bak_data_dir = os.path.join(header_dir, "fix_bak_data")
    preprocessed_dir = os.path.join(header_dir, "test_data_preprocessed")
    ori_img_path = './fake_originImages_add_wartermask'
    roi_img_path = './template_imgs/chusai_watermask_template.jpg'
    roi_rematch_img_path = "./template_imgs/fusai_watermask_template.jpg"
    csv_label = "./origin_data_csv_all.csv"
    base_template = "./template_imgs/origin_img_location_marker_template.jpg"
    for sub_dir in dirs:
        if not os.path.exists(os.path.join(header_dir, sub_dir)):
            os.makedirs(os.path.join(header_dir, sub_dir))
    roi_img = cv2.imread(roi_img_path)
    roi_img_gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("template1.jpg", roi_img_gray[16:56, 20:186])
    ret, roi_img_bin = cv2.threshold(roi_img_gray, 175, 255, cv2.THRESH_BINARY)
    water_points = []
    width, height = roi_img_bin.shape[::-1]
    for w in range(width):
        for h in range(height):
            if roi_img_bin[h][w] == 0:
                water_points.append([h, w])
    tp_threshold = 0.2
    if not args.no_gen_data_chu: # 生成用于去水印的测试集
        print("running gen_data ....")
        gen_align_real_test_data()
    if not args.no_gan_test: # 启动训练好的gan模型去水印
        print("running gan_test ....")
        gan_gen_result()
    if not args.no_rec_img: # 将去好水印的图片恢复到原图,只是像素点还原 
        print("running rec_img ....")
        recover_origin_img()
        if rematch:
            ori_img_path = recover_image_dir 
    if not args.no_gen_data_fu:
        print("running gen_data_fu ....")
        roi_img_path = roi_rematch_img_path
        roi_img = cv2.imread(roi_img_path)
        roi_img_gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
        ret, roi_img_bin = cv2.threshold(roi_img_gray, 175, 255, cv2.THRESH_BINARY)
        water_points = []
        width, height = roi_img_bin.shape[::-1]
        for w in range(width):
            for h in range(height):
                if roi_img_bin[h][w] == 0:
                    water_points.append([h, w])
        test_data_dst_path = os.path.join(header_dir, "pixel_test_data_all_11_8_fu")
        gen_align_real_test_data(rematch=True)
    if not args.no_gan_test_rematch: # 启动训练好的gan模型去水印
        print("running gan_test_fu ....")
        pixel_mode = args.gan_fu
        gan_result_dir = os.path.join(header_dir, "gan_result_fu_dir")
        gan_gen_result(rematch = True)
    if not args.no_rec_img_rematch: # 将去好水印的图片恢复到原图,只是像素点还原 
        print("running rec_img_fu ....")
        recover_image_dir = os.path.join(header_dir, "recover_image_fu_dir")
        recover_origin_img(rematch = True)
    if not args.no_train_data: # 生成用于文字识别训练的数据
        print("running train_data ....")
        run_gen_train_data(csv_label, train_data_dir, base_template, recover_image_dir)
    if not args.no_test_data: # 生成用于文字识别测试的数据
        print("running test_data ....")
        run_gen_test_data(train_data_dir, base_template, recover_image_dir)
    if not args.no_fix_img: # 将地址和签发机关的数据拆分成一行
        print("running fix_data ....")
        fix_address_unit(train_data_dir, fix_bak_data_dir)
    if not args.no_preprocessed: # 图像预处理
        print("running preprocessed_data ....")
        preprocess_imgs(train_data_dir, preprocessed_dir)
