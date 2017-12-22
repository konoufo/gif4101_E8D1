from PIL import Image
import os
from numpy import random
import numpy as np
from fnmatch import fnmatch
from skimage.external import tifffile
from scipy.misc import imsave
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

def uint2float(img):
    ''' This function converts a numpy3d array to type float
    NOTE. It didn't work with .astype(numpy.float) only...
    
    :param img: A 3d numpy array
    
    :returns : A 3d numpy array converted to float
    '''
    a,b,c = img.shape
    imgVector = img.reshape(1, a*b*c).astype(np.float)
    return imgVector.reshape((a,b,c))

def float2uint(img):
    ''' This function converts a numpy3d array to type float
    NOTE. It didn't work with .astype(numpy.float) only...
    
    :param img: A 3d numpy array
    
    :returns : A 3d numpy array converted to float
    '''
    a,b,c = img.shape
    imgVector = img.reshape(1, a*b*c).astype(np.uint16)
    return imgVector.reshape((a,b,c))

def rot_flip(img):
    i = random.randint(0,3)
    if i == 0:
        img = np.fliplr(img)
    elif i == 1:
        img = np.flipud(img)
    j = random.randint(0,4)
    if j == 0:
        img = np.rot90(img,1,(1,2))
    elif i == 1:
        img = np.rot90(img,2,(1,2))
    elif i == 2:
        img = np.rot90(img,3,(1,2))
    return img

def crop(Path, im_input, height, width, k):
    im = (tifffile.imread(im_input))
    #im = Image.open(im_input)
    #im = im_input.load()
    im = uint2float(im)
    im = rot_flip(im)
    chan,imgheight, imgwidth = im.shape
    if chan == 2:
        for i in range(0,imgheight+30,height):
            for j in range(0,imgwidth+30,width):
                w = j - 50 - random.randint(0,15) 
                if w<0:
                    w=0
                width_rand = width - random.randint(0,50) 
                if width_rand<50:
                    width_rand =50
                h = i - 50 - random.randint(0,15)
                if h<0:
                    h=0
                height_rand = height - random.randint(0,50)
                if height_rand<50:
                    height_rand=50
                if w+width_rand > imgwidth:
                    #w = imgwidth-width_rand
                    w = imgwidth - 100
                    width_rand = 100
                if h+height_rand > imgheight:
                    #h = imgheight-height_rand
                    h = imgheight - 100
                    height_rand = 100
                #box = (w, h, w + width_rand, h + height_rand)
                a = im[:,h:h+height_rand,w:w+width_rand]
                #a = gaussian_filter(a,sigma=2) # si l'on veut appliquer un blur sur les images
                a = float2uint(a)            
                #tifffile.imsave('img.tif',a)
                #a.save(os.path.join(Path, "IMG-%s.tif" % k))
                a = rot_flip(a)
                tifffile.imsave('C:\\Users\\Arthur\\Desktop\\projet\\data6\\EXP7-24-11-segmented-%s.tif' % k, a)
                k +=1
    
    else:
        imgheight,imgwidth, chan = im.shape
        for i in range(0,imgheight,height):
            for j in range(0,imgwidth,width):
                w = j - 50 
                if w<0:
                    w=0
                width_rand = width - random.randint(0,50) 
                if width_rand<50:
                    width_rand =50
                h = i - 50
                if h<0:
                    h=0
                height_rand = height - random.randint(0,50)
                if height_rand<50:
                    height_rand=50
                if w+width_rand > imgwidth:
                    #w = imgwidth-width_rand
                    w = imgwidth - 100
                    width_rand = 100
                if h+height_rand > imgheight:
                    #h = imgheight-height_rand
                    h = imgheight - 100
                    height_rand = 100
                    #box = (w, h, w + width_rand, h + height_rand)
                    a = im[h:h+height_rand,w:w+width_rand,:]
                    #a = gaussian_filter(a,sigma=2) # si l'on veut appliquer un blur sur les images
                    a = float2uint(a)            
                    #tifffile.imsave('img.tif',a)
                    #a.save(os.path.join(Path, "IMG-%s.tif" % k))
                    a = rot_flip(a)
                    tifffile.imsave('C:\\Users\\Arthur\\Desktop\\projet\\data6\\EXP7-24-11-segmented-%s.tif' % k, a)
                    k +=1
    return k

   
    
 
 
if __name__ == '__main__':
    #root = "C:\\Users\\Arthur\\Desktop\\Projet détection de clusters\\Protéines A(RIM)-B(PSD)\\EXP3-15-10-2017"
    root = 'C:\\Users\\Arthur\\Desktop\\Projet détection de clusters\\Protéines A(RIM)-B(PSD)\\EXP7-24-11-2017'
    keepFormat = "*.tif"
    
    fileList, nameList = [],[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, keepFormat):
                fileList.append(os.path.join(path,name))
                nameList.append(name)
    k=0
    for it,file_ in enumerate(fileList):
        image = file_
        #image_obj = cv2.imread(image,-1)
        imgwidth, imgheight = Image.open(image).size
        x = random.randint(50,100)
        y = random.randint(50,100) 
        path = 'C:\\Users\\Arthur\\Desktop\\test'    
        k = crop(path,image, y, x, k )
   
    
    




