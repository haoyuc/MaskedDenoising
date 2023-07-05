from skimage.exposure import match_histograms
import cv2
import numpy as np

for i in range(1,56):
    if i % 10 > 5 or i % 10 < 1:
        continue
    path = f"/home/haoyuc/Research/BSR/KAIR/results/1_3/img_{str(i).zfill(3)}_SwinIR.png"
    ref  = f"/home/haoyuc/Research/BSR/KAIR/testsets/urban100/urban100_all_LR/img_{str(i).zfill(3)}_x1.png"
    print('='*20)
    print(path)
    print(ref)
    img1 = cv2.imread(ref)
    img2 = cv2.imread(path)
    matched = match_histograms(img2, img1).astype(np.uint8)
    save = f"/home/haoyuc/Research/BSR/KAIR/results/1_3_matched/img_{str(i).zfill(3)}_x1.png"
    cv2.imwrite(save, matched)



# path1='E:/Vscode/NewData/Paper3Data/visible-infrared/3/visible1.tif'  
# path2='E:/Vscode/NewData/Paper3Data/visible-infrared/3/infrared1.tif'
# img1=cv2.imread(path1)
# img2=cv2.imread(path2)
# matched =match_histograms(img2,img1)
# cv2.namedWindow('1',cv2.WINDOW_NORMAL)
# cv2.imshow('1',img1)
# cv2.namedWindow('2',cv2.WINDOW_NORMAL)
# cv2.imshow('2',img2)
# cv2.namedWindow('match_img12',cv2.WINDOW_NORMAL)
# cv2.imshow('match_img12',matched.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()