import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage,misc
import matplotlib.pyplot as plt
import sys
import itertools

worksize = 32
numpoints = 64

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
    gradient =  np.array((vergradient,horgradient))

    if processing == 'normalize':
        """Normalizing the  gradient"""
        gradnorm = 0.2*np.max(np.linalg.norm(gradient,axis = -1))
        gradient = gradient / gradnorm

    return gradient

def coordinate_matrix(n):
    """Making an array of coordinates"""
    xcoordinates = np.zeros((n,n))
    xcoordinates = xcoordinates + np.arange(0,worksize) #broadcasting trick
    ycoordinates = xcoordinates.T
    return np.array([ycoordinates,xcoordinates])


def line_contribution(p1,p2,alpha = 1):
    """Outputs the matrix with with to adjust the gradient after adding the line between p1 and p2"""

    adjust = np.zeros((worksize,worksize,2))

    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    coordinates = coordinate_matrix(worksize)
    numerator = np.sum(np.multiply(coordinates,np.reshape(np.array(((y2-y1,-(x2-x1)))),(2,1,1))),axis = 0) + x2*y1 - y2*x1
    dist_from_line = np.abs(numerator) * (1.0/np.sqrt((y2-y1)**2+(x2-x1)**2))
    xcontribution = (x2-x1)*(1/(alpha*dist_from_line+1))
    ycontribution = (y2-y1)*(1/(alpha*dist_from_line+1))


    return np.array((ycontribution,xcontribution))/np.sqrt((y2-y1)**2+(x2-x1)**2)




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

rimpoints = perimeter_points(worksize,numpoints)
pixels = get_image(r"C:\Users\bskau\github\LineDraw\Star.png",worksize)
gradient = get_gradient(pixels)


lines = []
for line in itertools.combinations(rimpoints.T,2):
    lines.append(line)
lines = np.array(lines)
def lineloss(endpoints,gradient):
    """Return the loss assigned to the line between the two points"""
    l = discrete_line(endpoints[0],endpoints[1])
    direction = endpoints[1]-endpoints[0]
    dperp = np.array((-direction[1],direction[0])) #Perpendicular vector to the direction
    dperp = dperp/np.linalg.norm(dperp)
    lpoints = gradient[:,l[0],l[1]]


    return -np.sum(np.abs(np.dot(dperp,lpoints)))





outpixels = np.zeros((worksize,worksize)) + 255

cutoff = 5#Maximal number of lines

pickedlines = np.zeros(cutoff)
oldlines = np.copy(lines)




def show_n_best_lines(lines,cutoff):
    losses = np.array(list(map(lambda l : lineloss(l,gradient),lines)))

    drawlinesindices = np.argpartition(losses,cutoff)[:cutoff]

    for i in range(0,cutoff):
        line = lines[drawlinesindices[i]]
        linepixels = discrete_line(line[0],line[1])
        outpixels[linepixels[0],linepixels[1]] = 0


    outimage = Image.fromarray(outpixels)
    outimage.show()

#show_n_best_lines(lines,cutoff)


def clip_at_zero(M,N):
    """M,N are two matrices of vectors, returns a matrix which is M if M,N have positive dot product, and 0 otherwise"""
    dotprod = np.sum(np.multiply(M,N),axis = 0)
    if (np.any(np.logical_and(dotprod <= 0, np.all(M == np.array((0,0)).reshape((2,1,1)), axis=0)))):
        print('clipping happened')
    return np.where(dotprod<=0,np.zeros_like(M),M)

def iterate_n_best_lines(lines,n,plot = False):
    gradient = get_gradient(pixels)
    for i in range(0,n):
        losses = np.array(list(map(lambda l : lineloss(l,gradient),lines)))
        currentindex = np.argmin(losses)
        print('Index of line:')
        print(currentindex)
        currentlineends = lines[currentindex]
        currentline = discrete_line(currentlineends[0],currentlineends[1])
        currentdirection = currentlineends[1]-currentlineends[0]
        currentdirection = currentdirection/np.linalg.norm(currentdirection)
        outpixels[currentline[0],currentline[1]] = 0

        contribution =  1*line_contribution(currentlineends[0],currentlineends[1])

        subtractgrad = gradient - contribution
        addgrad = gradient + contribution


        plt.show()
        if plot:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2,ncols = 2)
            ax1.quiver(gradient[1],gradient[0])
            ax1.title.set_text('gradient')
            ax2.quiver(contribution[1],contribution[0])
            ax2.title.set_text('contribution')
            ax3.quiver(addgrad[1],addgrad[0])
            ax3.title.set_text('addgrad')
            ax4.quiver(subtractgrad[1],subtractgrad[0])
            ax4.title.set_text('subtractgrad')

        subtractgrad = clip_at_zero(subtractgrad,gradient)
        addgrad = clip_at_zero(addgrad,gradient)

        gradnorms = np.array((np.linalg.norm(gradient,axis = 0),np.linalg.norm(subtractgrad,axis = 0),np.linalg.norm(addgrad,axis = 0)))
        #Only subtract if this lowers the norm of the vector
        gradient = np.choose(np.argmin(gradnorms,axis = 0),(gradient,subtractgrad,addgrad))

        outimage = Image.fromarray(outpixels)
        print('max gradient')
        maxgrad = np.max(np.linalg.norm(gradient,axis = 0))
        print(np.max(np.linalg.norm(gradient,axis = 0)))
        if maxgrad == 0:
            print('exiting')
            break
        outimage.show()
        print('Iteration number:')
        print(i)


iterate_n_best_lines(lines,3,plot = True)
