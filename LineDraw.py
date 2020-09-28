import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


worksize = 8
numpoints = 8

rimpointsx = np.sin(np.linspace(0,2*np.pi,num=numpoints,endpoint = False))
rimpointsy = np.cos(np.linspace(0,2*np.pi,num=numpoints,endpoint = False))

rimpoints = np.round((worksize/2)*np.array((rimpointsx,rimpointsy)).T) + (worksize/2,worksize/2)


# Preprocessing of the image
image = Image.open(r"C:\Users\bskau\github\LineDraw\Lenna.png")


newimage = image.convert('LA').resize((worksize,worksize))
newimage.save(r"C:\Users\bskau\github\LineDraw\LennaBW.png")

pixels = np.asarray(newimage)[:,:,0]#.reshape(8,8)



def sign(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    else :
        return -1


def discrete_line(p1,p2):
    """Given two points in a grid, returns a list of all grid cells meeting the line between the points"""

    numpixels = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) #Taxicap metric distance

    xline = np.round(np.linspace(p1[0],p2[0],numpixels))
    yline = np.round(np.linspace(p1[1],p2[1],numpixels))

    return np.array([xline,yline]).T

l = discrete_line((0,0),(5,-3))
print(l)
