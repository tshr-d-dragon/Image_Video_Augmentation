import albumentations as A
import shutil
from glob import glob 
import cv2
import matplotlib.pyplot as plt

imgs = glob('') # path to images
anns = glob('')  # path to annotations

for i,j in tqdm(zip(imgs, anns), total=1005):
  img = cv2.imread(i)[:,:,::-1]
  # plt.imshow(img)
  # plt.show()

  transform = A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.3), contrast_limit=(-0.2, 0.3),
                                         brightness_by_max=True, p=1.0, always_apply=True)
  RBC = transform(image=img)
  # plt.imshow(RBC['image'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_RBC.'+i.split('.')[1], RBC['image'][:,:,::-1])
  shutil.copy(j, j.split('.')[0]+'_RBC.'+j.split('.')[1])

  transform = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True, p=1.0)
  CJ = transform(image=img)
  # plt.imshow(CJ['image'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_CJ.'+i.split('.')[1], CJ['image'][:,:,::-1])
  shutil.copy(j, j.split('.')[0]+'_CJ.'+j.split('.')[1])

  # transform = A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=True, p=1.0)
  # HSV = transform(image=img)
  # # plt.imshow(HSV['image'])
  # # plt.show()
  # cv2.imwrite(i.split('.')[0]+'_HSV.'+i.split('.')[1], HSV['image'][:,:,::-1])
  # shutil.copy(j, j.split('.')[0]+'_HSV.'+j.split('.')[1])

  transform = A.RandomBrightness(limit=(0.15, 0.3), always_apply=True, p=1.0)
  RB = transform(image=img)
  # plt.imshow(RB['image'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_RB.'+i.split('.')[1], RB['image'][:,:,::-1])
  shutil.copy(j, j.split('.')[0]+'_RB.'+j.split('.')[1])

  transform = A.RandomBrightness(limit=(-0.25, -0.15), always_apply=True, p=1.0)
  RB2 = transform(image=img)
  # plt.imshow(RB2['image'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_RB2.'+i.split('.')[1], RB2['image'][:,:,::-1])
  shutil.copy(j, j.split('.')[0]+'_RB2.'+j.split('.')[1])

  transform = A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True, p=1.0)
  Sharpen = transform(image=img)
  # plt.imshow(Sharpen['image'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_Sharpen.'+i.split('.')[1], Sharpen['image'][:,:,::-1])
  shutil.copy(j, j.split('.')[0]+'_Sharpen.'+j.split('.')[1])

  transform = A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.35, alpha_coef=0.08, always_apply=True, p=1.0)
  RF = transform(image=img)
  # plt.imshow(RF['image'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_RF.'+i.split('.')[1], RF['image'][:,:,::-1])
  shutil.copy(j, j.split('.')[0]+'_RF.'+j.split('.')[1])
