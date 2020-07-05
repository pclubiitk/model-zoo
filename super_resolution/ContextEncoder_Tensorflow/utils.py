import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def mask_randomly(args,imgs):
    
    y1 = np.random.randint(0, imgs.shape[1] - args.mask_height, imgs.shape[0])
    y2 = y1 + args.mask_height
    x1 = np.random.randint(0, imgs.shape[2] - args.mask_width, imgs.shape[0])
    x2 = x1 + args.mask_width

    masked_imgs = np.empty_like(imgs)
    missing_parts = np.empty((imgs.shape[0], args.mask_height, args.mask_width, imgs.shape[-1]))
    for i, img in enumerate(imgs):
        masked_img = np.asarray(img).copy()
        _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
        missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
        masked_img[_y1:_y2, _x1:_x2, :] = 1
        masked_imgs[i] = masked_img

    return masked_imgs, missing_parts, (y1, y2, x1, x2)
  

def sample_images(args,count, imgs):
        r, c = 3, args.num_img

        masked_imgs, missing_parts, (y1, y2, x1, x2) = mask_randomly(args,imgs)
        gen_missing = args.gen.predict(masked_imgs)

        imgs = 0.5 * imgs + 0.5
        masked_imgs = 0.5 * masked_imgs + 0.5
        gen_missing = 0.5 * gen_missing + 0.5

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0,i].imshow(imgs[i, :,:])
            axs[0,i].axis('off')
            axs[1,i].imshow(masked_imgs[i, :,:])
            axs[1,i].axis('off')
            filled_in = imgs[i].copy()
            filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
            axs[2,i].imshow(filled_in)
            axs[2,i].axis('off')

        fig.savefig(f"{count}.png")
        plt.close()

