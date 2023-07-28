import random
import albumentations as A
import cv2
from glob import glob

fps = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
vids = glob('') # Write the path for the videos files

for i in tqdm(vids):
  cap = cv2.VideoCapture(i)
  if (cap.isOpened()== False):
    print(i)

  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))

  size = (frame_width, frame_height)
  result0 = cv2.VideoWriter(f"/content/Dataset/{i.split('/')[-1][:-4]}_RBp.mp4",# fourcc,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          random.choice(fps), size)

  result1 = cv2.VideoWriter(f"/content/Dataset/{i.split('/')[-1][:-4]}_RBn.mp4",# fourcc,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          random.choice(fps), size)

  result2 = cv2.VideoWriter(f"/content/Dataset/{i.split('/')[-1][:-4]}_Sharpen.mp4",# fourcc,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          random.choice(fps), size)

  result3 = cv2.VideoWriter(f"/content/Dataset/{i.split('/')[-1][:-4]}_HSV.mp4",# fourcc,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          random.choice(fps), size)

  result4 = cv2.VideoWriter(f"/content/Dataset/{i.split('/')[-1][:-4]}_CJ.mp4",# fourcc,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          random.choice(fps), size)

  result5 = cv2.VideoWriter(f"/content/Dataset/{i.split('/')[-1][:-4]}_RBCp.mp4",# fourcc,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          random.choice(fps), size)

  result6 = cv2.VideoWriter(f"/content/Dataset/{i.split('/')[-1][:-4]}_RBCn.mp4",# fourcc,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          random.choice(fps), size)

  result7 = cv2.VideoWriter(f"/content/Dataset/{i.split('/')[-1][:-4]}_CS.mp4",# fourcc,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          random.choice(fps), size)

  result8 = cv2.VideoWriter(f"/content/Dataset/{i.split('/')[-1][:-4]}_RF.mp4",# fourcc,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          random.choice(fps), size)

  result9 = cv2.VideoWriter(f"/content/Dataset/{i.split('/')[-1][:-4]}_HF.mp4",# fourcc,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          random.choice(fps), size)

  while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break
    frame = frame[:,:,::-1]

    result0.write(A.RandomBrightness(limit=(0.25, 0.3), p=1.0, always_apply=True)(image=frame)['image'][:,:,::-1])

    result1.write(A.RandomBrightness(limit=(-0.2, -0.15), p=1.0, always_apply=True)(image=frame)['image'][:,:,::-1])

    result2.write(A.Sharpen(alpha=(0.35, 0.5), lightness=(0.8, 1.0), p=1.0, always_apply=True)(image=frame)['image'][:,:,::-1])

    result3.write(A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=True, p=1.0)(image=frame)['image'][:,:,::-1])

    result4.write(A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25, always_apply=True, p=1.0)(image=frame)['image'][:,:,::-1])

    result5.write(A.RandomBrightnessContrast(brightness_limit=(0.25, 0.3), contrast_limit=(0.25, 0.3),
                                            brightness_by_max=True, p=1.0, always_apply=True)(image=frame)['image'][:,:,::-1])

    result6.write(A.RandomBrightnessContrast(brightness_limit=(-0.25, -0.2), contrast_limit=(-0.25, -0.2),
                                            brightness_by_max=True, p=1.0, always_apply=True)(image=frame)['image'][:,:,::-1])

    result7.write(A.ChannelShuffle(always_apply=True, p=1.0)(image=frame)['image'][:,:,::-1])

    result8.write(A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.35, alpha_coef=0.08, always_apply=True, p=1.0)(image=frame)['image'][:,:,::-1])

    result9.write(A.HorizontalFlip(always_apply=True, p=1.0)(image=frame)['image'][:,:,::-1])

  cap.release()
  result0.release()
  result1.release()
  result2.release()
  result3.release()
  result4.release()
  result5.release()
  result6.release()
  result7.release()
  result8.release()
  result9.release()
  cv2.destroyAllWindows()