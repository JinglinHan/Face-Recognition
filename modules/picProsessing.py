'''
将所有照片统一大小
'''
import os
from PIL import Image
import cv2
import numpy as np
# import matplotlib.pyplot as plt


class picProcessor():
    def __init__(self, ID_name, ID, path):
        self.ID_name = ID_name
        self.ID = ID
        self.path = path

    def generateLabel(self):
        size = []
        x_dirs = os.listdir(self.path)
        y = []  # y保存了每一张照片的标签
        for x_file in x_dirs:
            im = Image.open(self.path + x_file)
            size.append(im.size)
            if (len(x_file)) < 9:
                y.append(0)
            else:
                for j in range(len(self.ID)):
                    if (x_file[:9] == self.ID[j][:9]):
                        y.append(j)
                    continue
        return y

    def generateData(self):
        x = np.zeros(shape=(1510, 9216))
        x_dirs = os.listdir(self.path)
        for i in range(len(x_dirs)):
            img = Image.open(self.path + x_dirs[i])
            im_grey = img.convert("L")
            new_img = im_grey.resize((72, 128), Image.ANTIALIAS)  # 采用了图片长宽平均值

            im_mat = np.matrix(new_img)
            im_mat = im_mat.reshape((1, 72 * 128))
            #    new_img.save('C:\\Users\\w\\Desktop\\PR_project\\dataReshape\\' + x_file)
            x[i] = im_mat
        return x





# size2 = list(set(size))
# # print(size2)                #[(544, 720), (464, 960), (544, 784), (544, 960), (544, 944), (540, 960), (448, 960)]
# # print(len(size2))
# a,b,c,d = 0,0,0,0
#
# for i in range(len(size)):
#     a +=size[i][0]
#     b +=size[i][1]
# for i in range(len(size2)):
#     c +=size2[i][0]
#     d +=size2[i][1]
# print(a/len(size))
# print(b/len(size))
# print(c/len(size2))
# print(d/len(size2))

# 537.7486772486773
# 939.1746031746031
# 518.2857142857143
# 898.2857142857143
#    print(round((size2[i][1]/size2[i][0]),2))
    # 1.32
    # 2.07
    # 1.44
    # 1.76
    # 1.74
    # 1.78
    # 2.14

#(448,720)




