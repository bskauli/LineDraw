import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage,misc
import matplotlib.pyplot as plt
import sys
import itertools

worksize =32
numpoints = 32

rimpointsx = np.sin(np.linspace(0,2*np.pi,num=numpoints,endpoint = False)) + 1
rimpointsy = np.cos(np.linspace(0,2*np.pi,num=numpoints,endpoint = False)) + 1

rimpoints = np.round((((worksize-1)/2))*np.array((rimpointsx,rimpointsy)).T)
rimpoints = rimpoints.astype(int)

# Preprocessing of the image
image = Image.open(r"C:\Users\bskau\github\LineDraw\Circle.png")

#image = image.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius = 5))
#newimage = image.convert('LA').resize((worksize+2,worksize+2))
#newimage.save(r"C:\Users\bskau\github\LineDraw\LennaBW.png")


newimage = image.resize((worksize,worksize)).convert('LA')
pixels = np.asarray(newimage,dtype = np.float32)[:,:,0]


horgradient = ndimage.sobel(pixels, axis = 1)
#horgradient = 255*horgradient/np.max(horgradient)

vergradient = ndimage.sobel(pixels, axis = 0)
#vergradient = 255*vergradient/np.max(vergradient)
#horgradient = ndimage.convolve(pixels,np.array([[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]))#,np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))

#horgradient = ndimage.gaussian_filter(horgradient,sigma = 5)
#vergradient = ndimage.gaussian_filter(vergradient,sigma = 5)

"""Making an array of coordinates, used to update the gradients after picking the best line"""

xcoordinates = np.zeros((worksize,worksize))
xcoordinates = xcoordinates + np.arange(0,worksize)

ycoordinates = xcoordinates.T

coordinates = np.swapaxes(np.swapaxes(np.array([ycoordinates,xcoordinates]),0,1),1,2)



fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side


ax1.imshow(horgradient)
ax2.imshow(vergradient)
#plt.show()

gradient = np.swapaxes(np.swapaxes(np.array([vergradient,horgradient]),0,1),1,2) #Transposing to correct shape
#plt.quiver(range(0,worksize),range(0,worksize),gradient[:,:,1],gradient[:,:,0])
#plt.show()

"""Normalizing the  gradient"""
gradnorm = 0.2*np.max(np.linalg.norm(gradient,axis = -1))


gradient = gradient / gradnorm



def line_contribution(p1,p2):
    """Outputs the matrix with with to adjust the gradient after adding the line between p1 and p2"""

    adjust = np.zeros((worksize,worksize,2))

    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]


    dist_from_line = np.abs(np.dot(coordinates,np.array((-(x2-x1),y2-y1))) + x2*y1 - y2*x1)
    dist_from_line = dist_from_line* (1.0/np.sqrt((y2-y1)**2+(x2-x1)**2))

    xcontribution = (x2-x1)*(1/(10*dist_from_line+1))
    ycontribution = (y2-y1)*(1/(10*dist_from_line+1))


    return np.swapaxes(np.swapaxes(np.array((xcontribution,ycontribution)),0,1),1,2)/np.sqrt((y2-y1)**2+(x2-x1)**2)



def sign(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    else :
        return -1


def discrete_line(p1,p2,alg = 'homebrew'):

    """
    Given two points in a grid, returns a list of all grid cells meeting the line between the points
    """
    if alg=='homebrew':
        numpixels = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) + 1 #Taxicap metric distance

        xline = np.rint(np.linspace(p1[0],p2[0],numpixels)).astype(int)
        yline = np.rint(np.linspace(p1[1],p2[1],numpixels)).astype(int)

        return np.array([xline,yline]).T

    elif alg == 'Bresenham':
        raise Exception
    elif alg == 'Wu':
        raise Exception


l = discrete_line((0,0),(5,3))

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
    lpoints = gradient[l[:,0],l[:,1]]

    return -np.sum(np.abs(np.dot(lpoints,dperp)))




#lossendslines = list(map((lambda e : (lineloss(e),e,discrete_line(e[0],e[1]))),lines))



outpixels = np.zeros((worksize,worksize)) + 255

cutoff = 50

for i in range(0,cutoff):

    plt.quiver(range(0,worksize),range(0,worksize),gradient[:,:,1],gradient[:,:,0])
    plt.show()
    losses = np.array(list(map(lineloss,lines)))
    minlineindex = np.argmin(losses)
    minline = lines[minlineindex]
    minlinepixels = discrete_line(minline[0],minline[1])
    outpixels[minlinepixels[:,0],minlinepixels[:,1]] = 0
    adjustment = line_contribution(minline[0],minline[1])
    gradient = np.minimum(gradient + adjustment,gradient - adjustment)
    print(i)


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
