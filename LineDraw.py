import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage,misc
import matplotlib.pyplot as plt
import sys
import itertools

worksize =8
numpoints = 4

def perimeter_points(d,n,type = 'int'):
    """Returns n points evenly spaced along the perimeter of a circle of diameter d centered at the origin,
       if type = 'int' the coordinates are rounded to the neares integer"""
    rimpointsx = np.sin(np.linspace(0,2*np.pi,num=n,endpoint = False)) + 1
    rimpointsy = np.cos(np.linspace(0,2*np.pi,num=n,endpoint = False)) + 1
    rimpoints = (((d-1)/2))*np.array([rimpointsy,rimpointsx])
    if type == 'int':
        rimpoints = np.round(rimpoints)
        rimpoints = rimpoints.astype(int)
    return rimpoints

print(perimeter_points(8,4))


# Preprocessing of the image
def get_image(filepath,size):
    """returns the image specified as a numpy array in grayscale
    Images cropped to squares
    """
    image = Image.open(filepath)
    newimage = image.resize((size,size)).convert('LA')
    pixels = np.asarray(newimage,dtype = np.float32)[:,:,0]
    return  pixels

def get_gradient(pixels,processing = 'normalize'):
    """returns the gradient of an image, and does basic preprocessing"""
    horgradient = ndimage.sobel(pixels, axis = 1)
    vergradient = ndimage.sobel(pixels, axis = 0)
    return np.array((horgradient,vergradient))

    if processing == 'normalize':
        """Normalizing the  gradient"""
        gradnorm = 0.2*np.max(np.linalg.norm(gradient,axis = -1))
        gradient = gradient / gradnorm


def coordinate_matrix(n):
    """Making an array of coordinates"""
    xcoordinates = np.zeros((n,n))
    xcoordinates = xcoordinates + np.arange(0,worksize) #broadcasting trick
    ycoordinates = xcoordinates.T
    return np.array([ycoordinates,xcoordinates])


def line_contribution(p1,p2):
    """Outputs the matrix with with to adjust the gradient after adding the line between p1 and p2"""

    adjust = np.zeros((worksize,worksize,2))

    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    coordinates = coordinate_matrix(worksize)
    print(x2-x1)
    numerator = np.sum(np.multiply(coordinates,np.reshape(np.array(((-(x2-x1),y2-y1))),(2,1,1))),axis = 0) + x2*y1 - y2*x1
    dist_from_line = numerator * (1.0/np.sqrt((y2-y1)**2+(x2-x1)**2))
    xcontribution = (x2-x1)*(1/(10*dist_from_line+1))
    ycontribution = (y2-y1)*(1/(10*dist_from_line+1))


    return np.array((xcontribution,ycontribution))/np.sqrt((y2-y1)**2+(x2-x1)**2)




def discrete_line(p1,p2,alg = 'homebrew'):

    """
    Given two points in a grid, returns a list of all grid cells meeting the line between the points
    """
    if alg=='homebrew':
        numpixels = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) + 1 #Taxicap metric distance

        xline = np.rint(np.linspace(p1[0],p2[0],numpixels)).astype(int)
        yline = np.rint(np.linspace(p1[1],p2[1],numpixels)).astype(int)

        return np.array([xline,yline])

    elif alg == 'Bresenham':
        raise Exception
    elif alg == 'Wu':
        raise Exception


l = discrete_line((0,0),(5,3))

rimpoints = perimeter_points(worksize,numpoints)
pixels = get_image(r"C:\Users\bskau\github\LineDraw\VertLine.png",worksize)
gradient = get_gradient(pixels)
print(gradient.shape)

lines = []
for line in itertools.combinations(rimpoints,2):
    lines.append(line)
lines = np.array(lines)
def lineloss(endpoints):
    """Return the loss assigned to the line between the two points"""
    l = discrete_line(endpoints[0],endpoints[1])
    direction = endpoints[1]-endpoints[0]
    dperp = np.array((-direction[1],direction[0])) #Perpendicular vector to the direction
    dperp = dperp/np.linalg.norm(dperp)
    lpoints = gradient[:,l[0],l[1]]


    return -np.sum(np.abs(np.dot(dperp,lpoints)))

print(lineloss(lines[0]))

outpixels = np.zeros((worksize,worksize)) + 255

cutoff = 10#Maximal number of lines

pickedlines = np.zeros(cutoff)
oldlines = np.copy(lines)

losses = np.array(list(map(lineloss,lines)))


for i in range(0,cutoff):

    minlineindex = np.argmin(losses)
    pickedlines[i] = minlineindex
    minline = lines[minlineindex]
    minlinepixels = discrete_line(minline[0],minline[1])
    outpixels[minlinepixels[:,0],minlinepixels[:,1]] = 0
    adjustment = line_contribution(minline[0],minline[1])


    #conditions = np.linalg.norm(gradient - adjustment,axis=-1)<np.linalg.norm(gradient,axis=-1)
    #conditions = np.repeat(conditions[:, :, np.newaxis], 2, axis=2)
    #gradient = np.where(conditions,gradient - adjustment,gradient)
    #conditions = np.linalg.norm(gradient + adjustment,axis=-1)<np.linalg.norm(gradient,axis=-1)
    #conditions = np.repeat(conditions[:, :, np.newaxis], 2, axis=2)
    #gradient = np.where(conditions,gradient + adjustment,gradient)
    #print(np.all(oldgradient==gradient))
    #print('---')
    #print(lines.shape)
    #lines = np.delete(lines,minlineindex,axis = 0)
    #print(lines.shape)

outimage = Image.fromarray(outpixels)
outimage.show()


sys.exit()

threshold = np.min(losses)/2.3
#print(losses<np.min(losses)/2)
#print(np.min(losses))
#print(lines.shape)
drawlines = lines[losses <= threshold]




#print(outpixels)
for i in range(0,len(drawlines)):
    currentline = discrete_line(drawlines[i][0],drawlines[i][1])
    outpixels[currentline[:,0],currentline[:,1]] = 0



# for i in range(0,cutoff):
#     currentline = lossendslines[i][2]
#     outpixels[currentline[:,0],currentline[:,1]] = 0
#     # if i > 0 and i % 50 == 0:
#     #     print(i)
#     #     command = input()
#     #     if command == "exit":
    #         break


outimage = Image.fromarray(outpixels)
outimage.show()





#outimage.save(r"C:\Users\bskau\github\LineDraw\LennaLine.png")
