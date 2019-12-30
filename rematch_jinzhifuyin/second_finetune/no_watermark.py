import cv2
import numpy as np
import os
import random
import copy
from multiprocessing import Pool
import sys
import argparse

class WatermarkRemove(object):
    def __init__(self,image_file,template_file,demo_file):
        '''
        加载相关的图像
        :param image_file:加载原始的图像
        :param template_file: 加载模板的图像
        :param demo_file: 用来做mask的图像
        '''
        self.origin_image=cv2.imread(image_file)
        self.origin_template=cv2.imread(template_file)
        self.image_gray=cv2.cvtColor(self.origin_image,cv2.COLOR_BGR2GRAY)
        self.template_gray=cv2.cvtColor(self.origin_template,cv2.COLOR_BGR2GRAY)
        self.demo_image=cv2.imread(demo_file)
        self.demo_gray=cv2.cvtColor(self.demo_image,cv2.COLOR_BGR2GRAY)
        self.all_pixels_value=None
        self.light_times=None               #对与全局图像来说亮度要放大的倍数
        self.roi_coordinate=None
        self.back_values=None               #背景填充的颜色像素值
        self.threshold=None                 #在使用二值化进行水印处理的时候需要使用动态的阈值

    def image_reverse(self,image):
        image_reverse=255-image
        return image_reverse

    def find_RoI(self,template, Origin_Image):
        '''
        根据模板匹配的方式直接得到对应的RoI区域
        :param template:
        :param Origin_Image:
        :return:
        '''
        method = cv2.TM_CCOEFF
        # Apply template Matching
        res = cv2.matchTemplate(Origin_Image, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            x, y = min_loc
        else:
            x, y = max_loc
        self.roi_coordinate=[x,y,x+template.shape[1],y+template.shape[0]]               # 保存原始图片中带水印区域的坐标
        self.roi=self.image_gray[y:y+template.shape[0],x:x+template.shape[1]]           # 通过切片操作将原始图片中带水印的区域保存在self.roi变量中
        # return x, y, x + template.shape[1], y + template.shape[0]

    def calculate_times(self):
        '''
        计算调整图像背景色时需要放大或是缩小的倍数
        :return:
        '''
        self.all_pixels_value=0
        for i in range(self.image_gray.shape[0]):
            for j in range(self.image_gray.shape[1]):
                self.all_pixels_value+=self.image_gray[i][j]
        self.light_times=self.all_pixels_value/16581827

    # 判断水印的位置，给出是否在正面有文字区域的判断
    def overlap_(self):
        '''
        根据RoI的位置判断是否在身份证的空白处，如果在，则直接进行填充
        否则再根据roi的像素情况选择对水印处理的方式
        :return:
        '''
        x1, y1, x2, y2 = self.roi_coordinate[0], self.roi_coordinate[1], self.roi_coordinate[2], self.roi_coordinate[3]
        #接下来判断是否在可以直接去除的区域内：
        if (x1>130 and y2<73 and x1<212):
            return True
        elif (x1>210 and y2<104 and x1<218):
            return True
        elif (x1>217 and y2<134):
            return True
        elif (x1>121 and y1>151 and y2<218):
            return True
        else:
            return False

    def calculate_background(self):
        '''
        计算输入图像右上角4*4block内的平均值，作为填充的背景色，使用灰度图进行计算
        :return:
        '''
        x1,y1,x2,y2=self.origin_image.shape[1]-4,0,self.origin_image.shape[1],4
        back_value=0
        for i in range(4):
            for j in range(4):
                back_value+=self.image_gray[i][j+x1]
        back_value=(back_value/16).astype(np.uint8)
        # return back_value
        self.back_values=back_value

    def choice_make(self):
        '''
        计算特定区域内的最小值的像素值的和，之后根据特定区域内的像素值和template内最小像素值的对比来选择对应的处理水印的方式
        :return:
        '''
        # 计算身份证正面文字部分的最小像素值的平均值
        x1,y1,x2,y2=73,138,208,174
        roi=self.image_gray[138:174,73:208]
        pixels_value=roi.min(axis=0).sum()/135          #计算水印所在区域的最小像素的总和
        # print(pixels_value.sum()/135)
        # 计算水印部分的最小像素值的平均数
        roi_pixels_value=self.roi.min(axis=0).sum()/177

        # print(roi_pixels_value.sum()/177)
        # 根据上述计算出来二者的差值来选择使用去水印的方式
        print(np.abs(pixels_value-roi_pixels_value))
        if np.abs(pixels_value-roi_pixels_value)>3:
            #如果最小值的差大于20，则使用二值化的方式进行处理
            self.threshold=roi_pixels_value.astype(np.uint8)
            self.watermark_remove_color_threshold()
            print('color_threshold')
        else:
            self.threshold=roi_pixels_value.astype(np.uint8)
            print('theory_based')
            self.watermark_remove_theory_based()

    def watermark_remove_theory_based(self):
        '''
        使用基于水印生成原理的方式来反向去水印
        计算动态阈值的时候，不能通过计算最大值或是最小值的方式来得到动态阈值，根据实际测试得出的结论
        目前想到的方法是计算两个对角上的
        :return:
        '''
        #先将水印所在区域转换为黑底白字
        self.roi=self.image_reverse(self.roi)
        intermediate_roi=(self.roi.astype(float)-0.85*self.demo_gray)/0.9
        intermediate_roi[intermediate_roi > 255] = 255
        intermediate_roi[intermediate_roi < 0] = 0
        dst = intermediate_roi.astype(np.uint8)
        dst = self.image_reverse(dst)
        #二值化的阈值也需要使动态的，但是这里需要重新计算经过处理之后roi区域中最大值像素的平均值
        #经过测试，这里需要使用平均的像素值来进行计算，如果使用最大像素值或是最小像素值都是无法得到很理想的效果
        threshold=np.median(dst,axis=0).sum()
        print(threshold)
        threshold=threshold/dst.shape[1]
        print(threshold)
        cv2.imshow('roi',dst)
        cv2.waitKey(0)
        print(self.back_values)
        _,dst=cv2.threshold(dst,threshold-50,self.back_values,cv2.THRESH_BINARY)
        cv2.imshow('dst',dst)
        x1, y1, x2, y2 = self.roi_coordinate[0], self.roi_coordinate[1], self.roi_coordinate[2], self.roi_coordinate[3]

        self.image_gray[y1:y2,x1:x2]=dst            #添加到原始图像中
        cv2.imshow('final',self.image_gray)
        cv2.waitKey(0)

    def RoI_padding(self):
        '''
        如果水印的位置在不需要处理的地方，直接对水印所在的位置像素值用背景色进行填充
        :return:
        '''
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                self.roi[i][j]=self.back_values
        x1, y1, x2, y2 = self.roi_coordinate[0], self.roi_coordinate[1], self.roi_coordinate[2], self.roi_coordinate[3]
        self.image_gray[y1:y2,x1:x2]=self.roi
        cv2.imshow('final',self.image_gray)
        cv2.waitKey(0)

    def watermark_remove_color_threshold(self):
        '''
        直接根据阈值进行二值化处理，将背景去掉。
        :param template_file: 模板的文件名
        :param image_file: 原始图像名
        :return:
        '''
        x1,y1,x2,y2=self.roi_coordinate[0],self.roi_coordinate[1],self.roi_coordinate[2],self.roi_coordinate[3]   #得到对应的RoI区域
        #接下来进行图像处理
        print(self.threshold)
        _,self.mask=cv2.threshold(self.roi,self.threshold+13,255,cv2.THRESH_BINARY)
        # self.mask=self.image_reverse(self.mask)
        cv2.imshow('mask',self.mask)
        cv2.waitKey(0)
        _,self.roi_processed=cv2.threshold(src=self.mask,thresh=0,maxval=self.back_values,type=cv2.THRESH_BINARY)
        cv2.imshow('roi_processed',self.roi_processed)
        cv2.waitKey(0)
        self.image_gray[y1:y2,x1:x2]=self.roi_processed
        cv2.imshow('image_final',self.image_gray)
        cv2.waitKey(0)


class WatermarkBlender(WatermarkRemove):
    def __init__(self, image_file, template_file, demo_file):
        '''
        添加水印的程序，创建一个新的类WatermarkBlender，
        因为要用到上面根据原理去水印类中的方法，所以继承上面的WatermakRemove类，
        主要是为了使用一些成员变量，类内的方法只使用了根据水印模板匹配当前处理图片中水印位置，也就是下面要用到的find_RoI这个函数
        :param image_file: 待添加水印图片所在的文件夹
        :param template_file: 水印模板匹配使用的模板图片
        :param demo_file: WatermarkRemove类中使用到的参数，本类中使用不到
        '''
        super(WatermarkBlender, self).__init__(image_file, template_file, demo_file)

    def add_watermark(self, param, ratio):
        '''
        进行水印平移的程序
        param: 以列表的方式给出，列表的第一个参数是待处理的图片，第二个参数是图片保存的文件夹路径
        rario: 控制生成图片的数量
        '''
        name = param[0].split('.')[0].split('/')[-1]                            # 提取出图片的名称
        image_blender = cv2.imread(param[0])                                    # 通过opencv读取图片
        image_blender = cv2.cvtColor(image_blender,cv2.COLOR_BGR2GRAY)          # 将图片转换为灰度图格式
        self.find_RoI(self.template_gray, image_blender)						# 能够根据水印的模板找到对应原始图片中水印的位置，带水印的区域被存放在self.roi变量中，水印的坐标保存在self.
        index = param[0].split('.')[0].split('_')[-1]                           # 获取图片名称的最后一个字符，并由此判断当前正在处理的是图片的正面还是背面
        median_origin = np.median(self.roi, axis=0).sum() / self.roi.shape[0]   # 计算原始图片水印位置处像素值中位数的平均值，得到原始水印位置处的亮度情况，在之后的处理过程中要保持这个亮度值不变   
        # 处理身份证的正面
        if index == '1':
            cnt = 0
            for row in range(12):  # 列方向上处理的次数
                for col in range(9):  # 行方向上处理的次数
                    origin_content = image_blender[5 + row * 15: 5 + row * 15 + self.roi.shape[0],
                                     5 + col * 15: 5 + col * 15 + self.roi.shape[1]]				# 在空白位置处找到水印大小的区域，保存在origin_content变量中
                    for i in range(25):                                         # 为了最大可能模拟给出训练集上的水印和底片的模糊程度，在添加水印的过程中，设置不同程度的原始底片的透明度
                        trans_origin = i * 0.01 + 0.15					        # 数值通过事先自己的实验得出，在该设置下，能够较好地模拟水印和原始底片之间的模糊程度关系，制造出来的图片比较逼真
                        watermark_content_1 = origin_content * trans_origin + self.roi * 1.2        # 根据水印的公式，将当前图片中的水印区域和待处理区域以不同的比例进行融合

                        median_template_1 = np.median(watermark_content_1, axis=0).sum() / self.roi.shape[0]        # 计算融合后整个区域的亮度情况
                        watermark_content_1 = np.clip(watermark_content_1 - median_template_1 + median_origin - 10,	# 这个是动态的设置一下自己生成的水印和原始图片中的水印要保证亮度上相同
                                                      0, 255)                                                       # 同时对处理之后的图像进行范围的限制0-255
                        image_1 = cv2.imread(param[0])                          # 编程中忽视了python中变量copy的机制，所以采用了笨重的方法每次都读取原始的图片，可以用deepcopy来代替
                        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
                        image_1[5 + row * 15: 5 + row * 15 + self.roi.shape[0],
                        5 + col * 15: 5 + col * 15 + self.roi.shape[1]] = watermark_content_1       # 将正在处理的区域用融合后的图像来代替
                        if row * 10 + 128 < row * 15 + 5 + self.roi.shape[0]:					    # 接下来对融合后的周边区域进行裁切，得到包含融合区域在内的128*128分辨率大小的区域
                            row_end = row * 15 + 5 + self.roi.shape[0] + 10
                            row_start = row_end - 128
                        else:
                            row_end = row * 10 + 128
                            row_start = row * 10
                        if col * 10 + 128 < 5 + col * 15 + self.roi.shape[1]:
                            col_end = 5 + col * 15 + self.roi.shape[1] + 10
                            col_start = col_end - 128
                        else:
                            col_end = col * 10 + 128
                            col_start = col * 10
                        real_image = cv2.imread(param[0])                                           
                        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
                        real_image = real_image[row_start: row_end, col_start: col_end]
                        image_1 = image_1[row_start: row_end, col_start: col_end]                   # 得到原始底片中对应裁切区域的干净的，作为去水印结果的真实值
                        image_1 = np.hstack((image_1, real_image))                                  # 将添加水印的裁剪区域和原始底片的对应裁剪区域并排显示
                        if random.randint(0, 100) < ratio:                                              # 通过随机数在每张图片上随机生成不同模糊程度的图片，同时也控制生成图片的数量
                            cv2.imwrite(param[1] +name + '_%d' % (cnt) + '.jpg', image_1)
                            cnt += 1
                            print('writing image', name)
        if index == '0':                                                                            # 处理身份证的背面，和上面的程序相比，只有添加水印位置不同的区别。
            print('0')
            cnt = 0
            # 处理身份证的反面
            for row in range(3):  # 列方向上
                for col in range(23):  # 行方向上
                    origin_content = image_blender[5 + row * 15: 5 + row * 15 + self.roi.shape[0],
                                     5 + col * 15: 5 + col * 15 + self.roi.shape[1]]
                    print(origin_content.shape)
                    for i in range(25):
                        trans_origin = i * 0.01 + 0.15
                        watermark_content_1 = origin_content * trans_origin + self.roi * 1.2

                        median_template_1 = np.median(watermark_content_1, axis=0).sum() / self.roi.shape[0]
                        watermark_content_1 = np.clip(watermark_content_1 - median_template_1 + median_origin - 10,
                                                      0, 255)
                        image_1 = cv2.imread(param[0])
                        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
                        image_1[5 + row * 15: 5 + row * 15 + self.roi.shape[0],
                        5 + col * 15: 5 + col * 15 + self.roi.shape[1]] = watermark_content_1
                        if row * 10 + 128 < row * 15 + 5 + self.roi.shape[0]:
                            row_end = row * 15 + 5 + self.roi.shape[0] + 10
                            row_start = row_end - 128
                        else:
                            row_end = row * 10 + 128
                            row_start = row * 10
                        if col * 10 + 128 < 5 + col * 15 + self.roi.shape[1]:
                            col_end = 5 + col * 15 + self.roi.shape[1] + 10
                            col_start = col_end - 128
                        else:
                            col_end = col * 10 + 128
                            col_start = col * 10
                        real_image = cv2.imread(param[0])
                        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
                        real_image = real_image[row_start: row_end, col_start: col_end]
                        image_1 = image_1[row_start: row_end, col_start: col_end]
                        image_1 = np.hstack((image_1, real_image))
                        if random.randint(0, 100) < ratio:
                            cv2.imwrite(param[1] +name + '_%d' % (cnt) + '.jpg', image_1)
                            cnt += 1
                            print('writing image', name)
            for col in range(7):
                try:
                    origin_content = image_blender[192 + self.roi.shape[0] - 95: 192 + self.roi.shape[0],
                                    132 + col * 15: 132 + col * 15 + self.roi.shape[1]]
                    print(origin_content.shape)
                    for i in range(25):
                        trans_origin = i * 0.01 + 0.15
                        watermark_content_1 = origin_content * trans_origin + self.roi * 1.2

                        median_template_1 = np.median(watermark_content_1, axis=0).sum() / self.roi.shape[0]
                        watermark_content_1 = np.clip(watermark_content_1 - median_template_1 + median_origin - 10,
                                                      0, 255)
                        image_1 = cv2.imread(param[0])
                        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
                        image_1[192 + self.roi.shape[0] - 95 : 192 + self.roi.shape[0],
                                    132 + col * 15: 132 + col * 15 + self.roi.shape[1]] = watermark_content_1
                        row_start = 146
                        row_end = 274
                        col_start = 127 + col * 15
                        col_end = 255 + col * 15
                        real_image = cv2.imread(param[0])
                        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
                        real_image = real_image[row_start: row_end,
                                    col_start: col_end]
                        image_1 = image_1[row_start: row_end,
                                    col_start: col_end]
                        image_1 = np.hstack((image_1, real_image))
                        if random.randint(0, 100) < ratio:
                            cv2.imwrite(param[1] +name + '_%d' % (cnt) + '.jpg', image_1)
                            cnt += 1
                            print('writing image', name)
                except:
                    print('resolution wrong', name)

def parse_arguments(argv):                              # 读取外部的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_file', type=str, help='watermark template used for matching the watermark')
    parser.add_argument('--file_path', type=str, help='path to load the source images')
    parser.add_argument('--target_path', type=str, help='path to save the processed images')
    parser.add_argument('--ratio', type=int, help='control the amount of generated images')
    parser.add_argument('--demo_file', type=str, default='roi.jpg', help='demo file used for the original watermark remover')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_arguments(sys.argv[1:])
    template_file = args.template_file
    file_path = args.file_path
    target_path = args.target_path
    ratio = args.ratio
    demo_file = args.demo_file
    pool = Pool(processes=30)                           # 多进程处理的进程数，根据实际情况设定
    for file in os.listdir(file_path):
        image_file = file_path + file
        print(image_file)
        m = WatermarkBlender(image_file, template_file, demo_file)                      # 实例化一个对象
        pool.apply_async(m.add_watermark, ([image_file, target_path], ratio, ))    # 多进程进行添加水印的处理
    pool.close()
    pool.join()