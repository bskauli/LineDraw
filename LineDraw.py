import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage,misc
import matplotlib.pyplot as plt
import sys
import itertools

worksize = 512
numpoints = 128

rimpointsx = np.sin(np.linspace(0,2*np.pi,num=numpoints,endpoint = False)) + 1
rimpointsy = np.cos(np.linspace(0,2*np.pi,num=numpoints,endpoint = False)) + 1

rimpoints = np.round((((worksize-1)/2))*np.array((rimpointsx,rimpointsy)).T)
rimpoints = rimpoints.astype(int)

# Preprocessing of the image
image = Image.open(r"C:\Users\bskau\github\LineDraw\Cross.png")

#image = image.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius = 5))
#newimage = image.convert('LA').resize((worksize+2,worksize+2))
#newimage.save(r"C:\Users\bskau\github\LineDraw\LennaBW.png")


newimage = image.resize((worksize,worksize)).convert('LA')
pixels = np.asarray(newimage,dtype = np.float32)[:,:,0]
print(pixels.dtype)


horgradient = ndimage.sobel(pixels, axis = 1)
#horgradient = 255*horgradient/np.max(horgradient)

vergradient = ndimage.sobel(pixels, axis = 0)
#vergradient = 255*vergradient/np.max(vergradient)
#horgradient = ndimage.convolve(pixels,np.array([[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]))#,np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))

#horgradient = ndimage.gaussian_filter(horgradient,sigma = 5)
#vergradient = ndimage.gaussian_filter(vergradient,sigma = 5)

fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side


ax1.imshow(horgradient)
ax2.imshow(vergradient)
#plt.show()

gradient = np.swapaxes(np.swapaxes(np.array([vergradient,horgradient]),0,1),1,2) #Transposing to correct shape
plt.quiver(range(0,worksize),range(0,worksize),gradient[:,:,1],gradient[:,:,0])
#plt.show()

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
lines = np.array(lines)
def lineloss(endpoints):
    """Return the loss assigned to the line between the two points"""
    l = discrete_line(endpoints[0],endpoints[1])
    direction = endpoints[1]-endpoints[0]
    dperp = np.array((-direction[1],direction[0])) #Perpendicular vector to the direction
    lpoints = gradient[l[:,0],l[:,1]]

    return -np.sum(np.abs(np.dot(lpoints,dperp)))



losses = np.array(list(map(lineloss,lines)))

#lossendslines = list(map((lambda e : (lineloss(e),e,discrete_line(e[0],e[1]))),lines))


d = np.array([0,-15])


#l1 = lossendslines[1]
#print(l1[0])
#print(l1[1])
#print(l1[2])
#print(gradient.shape)
#print(np.sum(gradient[l1[2][:,0],l1[2][:,1]]))
#sys.exit()
#lossendslines.sort(key = lambda e : e[0])


cutoff = 100

threshold = np.min(losses)/10
#print(losses<np.min(losses)/2)
#print(np.min(losses))
#print(lines.shape)
drawlines = lines[losses <= threshold]



outpixels = np.zeros((worksize,worksize)) + 255
print(outpixels)
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
