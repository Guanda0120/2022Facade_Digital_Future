from PIL import Image
import numpy as np
import os

pic_path = r"D:\LiGD\NoahDF\DF_transferLearning\Horizon\0"
pic_list = os.listdir(pic_path)
new_pic_sz = 10
pic_array_container = []
for i in range(1000):
    pic = pic_list[i]
    tmp_pic_dir = os.path.join(pic_path,pic)
    tmp_pic = Image.open(tmp_pic_dir)
    tmp_pic = tmp_pic.resize((250,250))
    pic_array = np.asarray(tmp_pic)
    if pic_array.shape == (250,250,3):
        pic_array_container.append(pic_array)
pre_row = None
row_all = None
for j in range(30):
    for k in range(30):
        if k==0:
            pre_row = pic_array_container[j*30+k]
        else:
            pre_row = np.hstack((pre_row,pic_array_container[j*30+k]))

    if j == 0:
        row_all = pre_row
    else:
        row_all = np.vstack((row_all,pre_row))

new_img = Image.fromarray(np.uint8(row_all))
new_img.save('D:\\LiGD\\NoahDF\\DF_transferLearning\\1.jpg')