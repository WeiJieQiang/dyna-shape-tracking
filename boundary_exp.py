from image_class import *
import matplotlib.pyplot as plt

image = Image()

"""show the image in png"""
#plt.imshow(image.image_png[1:100,:,:])
#plt.show()


"""png to numpy array"""
image_np = image.png_to_np()


"""The fourth channel of the image is the white background with value 255, so we can get rid of the fourth channel 
and make the pic have only three channel"""
if image_np[:,:,3].min() ==255:
    image_np = image_np[:,:,0:3]

plt.imshow(image.image_mat)
#plt.show()

"""Fetch the boundary out of the image_mat, evaluated around 2 +- 0.1"""
bound_e = 2
err = 0.1
if np.argwhere(np.isnan(image.image_mat)).size == 0:
    boundary = ((image.image_mat >= bound_e-err)&(image.image_mat <= bound_e+err))
else:
    boundary = ((np.nan_to_num(image.image_mat) >= bound_e-err)&(np.nan_to_num(image.image_mat) <= bound_e+err))

plt.imshow(boundary)
plt.show()
