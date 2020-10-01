import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import itertools

worksize = 8
numpoints = 8

rimpointsx = np.sin(np.linspace(0,2*np.pi,num=numpoints,endpoint = False)) + 1
rimpointsy = np.cos(np.linspace(0,2*np.pi,num=numpoints,endpoint = False)) + 1

rimpoints = np.round((((worksize-1)/2))*np.array((rimpointsx,rimpointsy)).T)
rimpoints = rimpoints.astype(int)

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

    """
    Given two points in a grid, returns a list of all grid cells meeting the line between the points

    Current implementation is imperfect, as it skips a point on at least horizontal&vertical lines
    """

    numpixels = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) #Taxicap metric distance

    xline = np.rint(np.linspace(p1[0],p2[0],numpixels)).astype(int)
    yline = np.rint(np.linspace(p1[1],p2[1],numpixels)).astype(int)

    return np.array([xline,yline]).T

l = discrete_line((0,0),(5,3))

lines = []
for line in itertools.combinations(rimpoints,2):
    lines.append(line)

def lineweight(endpoints):
    """Return the weight assigned to the line between the two points"""
    l = discrete_line(endpoints[0],endpoints[1])
    return np.sum(pixels[l[:,0],l[:,1]])/len(l)

weights = np.array(list(map(lineweight,lines)))
lowestline = lines[np.argmin(weights)]
print(np.min(weights))
print(lowestline)

print(pixels)
print(pixels[lowestline[0][0],lowestline[0][1]])
