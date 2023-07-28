import albumentations as A
import cv2
from glob import glob
import matplotlib.pyplot as plt

images = glob('') # path to images
masks = glob('') # path to annotations

for i,j in tqdm(zip(images, masks)):
  
  img = cv2.imread(i)[:,:,::-1]
  # plt.imshow(img)
  # plt.show()

  mask = cv2.imread(j,0)
  # plt.imshow(mask)
  # plt.show()

  transform = A.Blur(blur_limit=4, p=1.0, always_apply=True)
  Blur = transform(image=img, mask=mask)
  # plt.imshow(HFlip['image'])
  # plt.show()
  # plt.imshow(HFlip['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_Blur.'+i.split('.')[1], Blur['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_Blur.'+j.split('.')[1], Blur['mask'])

  transform = A.Sharpen(alpha=(0.15, 0.3), lightness=(0.5, 1.0), p=1.0, always_apply=True)
  Sharpen = transform(image=img, mask=mask)
  # plt.imshow(HFlip['image'])
  # plt.show()
  # plt.imshow(HFlip['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_Sharpen.'+i.split('.')[1], Sharpen['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_Sharpen.'+j.split('.')[1], Sharpen['mask'])
  
  transform = A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.2, 0.2), 
                                         brightness_by_max=True, p=1.0, always_apply=True)
  RBC = transform(image=img, mask=mask)
  # plt.imshow(HFlip['image'])
  # plt.show()
  # plt.imshow(HFlip['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_RBC.'+i.split('.')[1], RBC['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_RBC.'+j.split('.')[1], RBC['mask'])

  transform = A.GridDistortion(num_steps=2, distort_limit=0.3, border_mode=2, p=1.0, always_apply=True)
  GD = transform(image=img, mask=mask)
  # plt.imshow(HFlip['image'])
  # plt.show()
  # plt.imshow(HFlip['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_GD.'+i.split('.')[1], GD['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_GD.'+j.split('.')[1], GD['mask'])

  transform = A.ElasticTransform(alpha=0.7, sigma=25, alpha_affine=25, border_mode=2, p=1.0, always_apply=True)
  ET = transform(image=img, mask=mask)
  # plt.imshow(HFlip['image'])
  # plt.show()
  # plt.imshow(HFlip['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_ET.'+i.split('.')[1], ET['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_ET.'+j.split('.')[1], ET['mask'])


  transform = A.HorizontalFlip(p=1.0, always_apply=True)
  HFlip = transform(image=img, mask=mask)
  # plt.imshow(HFlip['image'])
  # plt.show()
  # plt.imshow(HFlip['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_HFlip.'+i.split('.')[1], HFlip['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_HFlip.'+j.split('.')[1], HFlip['mask'])

  transform = A.VerticalFlip(p=1.0, always_apply=True)
  VFlip = transform(image=img, mask=mask)
  # plt.imshow(VFlip['image'])
  # plt.show()
  # plt.imshow(VFlip['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_VFlip.'+i.split('.')[1], VFlip['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_VFlip.'+j.split('.')[1], VFlip['mask'])

  transform = A.VerticalFlip(p=1.0, always_apply=True)
  VHFlip = transform(image=HFlip['image'], mask=HFlip['mask'])
  # plt.imshow(VHFlip['image'])
  # plt.show()
  # plt.imshow(VHFlip['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_VHFlip.'+i.split('.')[1], VHFlip['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_VHFlip.'+j.split('.')[1], VHFlip['mask'])

  transform = A.Rotate(p=1.0, always_apply=True)
  R = transform(image=img, mask=mask)
  # plt.imshow(R['image'])
  # plt.show()
  # plt.imshow(R['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_R.'+i.split('.')[1], R['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_R.'+j.split('.')[1], R['mask'])

  transform = A.ShiftScaleRotate(p=1.0, always_apply=True)
  SSR = transform(image=img, mask=mask)
  # plt.imshow(SSR['image'])
  # plt.show()
  # plt.imshow(SSR['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_SSR.'+i.split('.')[1], SSR['image'][:,:,::-1]) 
  cv2.imwrite(j.split('.')[0]+'_SSR.'+j.split('.')[1], SSR['mask']) 

  transform = A.ShiftScaleRotate(p=1.0, always_apply=True)
  VHFlipSSR = transform(image=VHFlip['image'], mask=VHFlip['mask'])
  # plt.imshow(VHFlipSSR['image'])
  # plt.show()
  # plt.imshow(VHFlipSSR['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_VHFlipSSR.'+i.split('.')[1], VHFlipSSR['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_VHFlipSSR.'+j.split('.')[1], VHFlipSSR['mask'])

  transform = A.SafeRotate(p=1.0, always_apply=True)
  SR = transform(image=img, mask=mask)
  # plt.imshow(SR['image'])
  # plt.show()
  # plt.imshow(SR['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_RSC.'+i.split('.')[1], SR['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_RSC.'+j.split('.')[1], SR['mask'])

  transform = A.SafeRotate(p=1.0, always_apply=True)
  VHFlipSR = transform(image=VHFlip['image'], mask=VHFlip['mask'])
  # plt.imshow(VHFlipSR['image'])
  # plt.show()
  # plt.imshow(VHFlipSR['mask'])
  # plt.show()
  cv2.imwrite(i.split('.')[0]+'_VHFlipSR.'+i.split('.')[1], VHFlipSR['image'][:,:,::-1])
  cv2.imwrite(j.split('.')[0]+'_VHFlipSR.'+j.split('.')[1], VHFlipSR['mask'])
