import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import itertools

worksize = 512
numpoints = 100

rimpointsx = np.sin(np.linspace(0,2*np.pi,num=numpoints,endpoint = False)) + 1
rimpointsy = np.cos(np.linspace(0,2*np.pi,num=numpoints,endpoint = False)) + 1

rimpoints = np.round((((worksize-1)/2))*np.array((rimpointsx,rimpointsy)).T)
rimpoints = rimpoints.astype(int)

# Preprocessing of the image
image = Image.open(r"C:\Users\bskau\github\LineDraw\Lenna.png")

image = image.filter(ImageFilter.FIND_EDGES)
newimage = image.convert('LA').resize((worksize+2,worksize+2))
newimage.save(r"C:\Users\bskau\github\LineDraw\LennaBW.png")

pixels = np.asarray(newimage)[1:-1,1:-1,0]#.reshape(8,8)

pixels = pixels*(-1)+255

def sign(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    else :
        return -1


def discrete_line(p1,p2):

    """
    Given two points in a grid, returns a list of all grid cells meeting the line between the points
    """

    numpixels = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) + 1 #Taxicap metric distance

    xline = np.rint(np.linspace(p1[0],p2[0],numpixels)).astype(int)
    yline = np.rint(np.linspace(p1[1],p2[1],numpixels)).astype(int)

    return np.array([xline,yline]).T

l = discrete_line((0,0),(5,3))

lines = []
for line in itertools.combinations(rimpoints,2):
    lines.append(line)

def lineloss(endpoints):
    """Return the loss assigned to the line between the two points"""
    l = discrete_line(endpoints[0],endpoints[1])
    lpoints = pixels[l[:,0],l[:,1]]
    return np.sum(lpoints**2)/len(l)



losses = list(map(lineloss,lines))

lossendslines = list(map((lambda e : (lineloss(e),e,discrete_line(e[0],e[1]))),lines))

lossendslines.sort(key = lambda e : e[0])

cutoff = 300
outpixels = np.zeros((worksize,worksize)) + 255

for i in range(0,cutoff):
    currentline = lossendslines[i][2]
    outpixels[currentline[:,0],currentline[:,1]] = 0

outimage = Image.fromarray(outpixels)
outimage.show()
#outimage.save(r"C:\Users\bskau\github\LineDraw\LennaLine.png")
