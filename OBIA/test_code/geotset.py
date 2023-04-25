import geopandas as gpd
import geopandas.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from OBIA.s3_features_match import shp2raster
from OBIA.s1_img_seg import write_img,read_img


# world = gpd.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# world.plot()
# plt.show()

# in_shp = r'E:\20230328_GEOBIA\extra_data\class_samples.shp'
# img_path = r'E:\20230328_GEOBIA\step1_segmentation'
# seg_img = os.path.join(img_path, 'test1.tif')
# ground_truth, classes = shp2raster(train_shp=in_shp, seg_img=seg_img, field='Id')
#
#
# im_width, im_height, im_proj, im_geotrans, im_data = read_img(seg_img)
# ground_truth  = ground_truth.transpose(1,0)
# write_img(filename='test.tif', im_proj=im_proj, im_geotrans=im_geotrans, im_data=ground_truth)
# print(ground_truth.shape)
# print(classes)
# np.repeat()


def test():
    a = 1
    b=2
    c=3
    return {'1':a,'2':b,"3":c}
l = test()
a,b,c = test()
print(a, b, c)
print(type(a))

a = np.array([[1,2,3,4],[4,5,6,7],[10,11,12,13,]])
print(a[[0,1]])

def question():
   
    pass

if __name__ == '__main__':
    pass
