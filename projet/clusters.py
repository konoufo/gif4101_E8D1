
import numpy
import os
import time
import csv

from matplotlib import pyplot

from skimage.external import tifffile
from skimage.feature import blob_doh,blob_log,blob_dog
import skimage

from scipy import optimize, interpolate

from fnmatch import fnmatch


def norm_img(img):
    ''' This function normalizes a stack image

    :param img: The numpy stack image

    :returns : A normalized stack image
    '''
    for chan in range(img.shape[0]):
        img[chan, :, :] = img[chan, :, :] - numpy.amin(img[chan, :, :])
    return img
    
    
def uint2float(img):
    ''' This function converts a numpy3d array to type float
    NOTE. It didn't work with .astype(numpy.float) only...
    
    :param img: A 3d numpy array
    
    :returns : A 3d numpy array converted to float
    '''
    a,b,c = img.shape
    imgVector = img.reshape(1, a*b*c).astype(numpy.float)
    return imgVector.reshape((a,b,c))


def blob_detection(img, percentile=95):
    ''' This function computes a blob detection on a numpy array.
    The blob detection is the "Difference of Gaussian". This function is
    from skimage. NOTE. Other detections may be used such as "Difference of Hessian"
    and "Laplacian of Gaussian". The parameters should the be changed.
    NOTE. Since the images all have ~ the same intensity we can set a threshold based
    on the intensity.
    
    :param img : A numpy array of the img
    
    :returns : A list of list of blobs for every image in the stack [x,y,radius]
    '''
    # threshold = numpy.percentile(img,percentile)
    threshold = 4
    return [blob_log(im,min_sigma=3,max_sigma=25,threshold=threshold,overlap=0.9) for im in img]


def dist(p1, p2):
    ''' This function computes the distance absolute distance between two points
    NOTE. It takes less time than to use the dist function
    
    :param p1: A (x,y) coordinate
    :param p2: A (x,y) coordiante
    
    :reutrns : The absolute distance between two points
    '''
    x1,y1 = p1
    x2,y2 = p2
    return (x1-x2)**2+(y1-y2)**2
    
    
def dist_list(point, blobList, nonzero=False):
    ''' This function computes the distance between a point and every other points in a list
    
    :param point: A tuple (x,y,r) where r won't be necessary
    :param blobList: A list of blobs
    
    :returns : The minimal distance and the other blob index
    '''
    y1,x1,r1 = point
    distance = []
    for i, blob in enumerate(blobList):
        y2,x2,r2 = blob
        distance.append([i, dist((x1,y1), (x2,y2))])
    try:
        if nonzero:
            return min((d,i) for i,d in distance if d > 0)
        else:
            return min((d,i) for i,d in distance)
    except ValueError:
        return (-1, -1)


def blob_distance(blobs, i, j):
    ''' This function computes the distance between every blob in lists of blobs
    and then returns the minimum value (nearest neighbor) of every blob
    NOTE. The distance is for the ith list of blobs computed against the jth list of blobs
    
    :param blobs: A list of two list of blobs
    
    :returns : A list of nearest neighbor and the index of the blob (dist, i)
    '''
    blob = blobs[i]
    distance = []
    for b in blob:
        distance.append(dist_list(b, blobs[j]))
    return distance
    
    
def blob_distance_same_image(blobs):
    ''' This function computes the distance between every blobs in a list of blobs
    and returns the nearest neighbor from a blob in the same list of blobs
    
    :param blobs: A list of list of blobs
    
    :returns : A list of list of nearest neighbor and the index of the blob (dist, i)
    '''
    nearestNeigh = []
    for blob in blobs:
        neigh = []
        for b in blob:
            neigh.append(dist_list(b,blob,nonzero=True))
        nearestNeigh.append(neigh)
    return nearestNeigh
    
    
def blob_neigh_number(parameters,blobs,d):
    ''' This function computes the distance between blobs in a blob list and if the distance is
    less than the specified distance the number of neigbord in the area increments by 1.
    
    :param parameters: A list of fit parameters
    :param blobs: A list of blobs containing every blobs of the parameters
    :param d: A minium distance (in pixels)
    
    :returns : A list of list of blobs with the number of neighbors
    '''
    neighNum = []
    for i,p,c in parameters:
        dx1,dy1 = p[1],p[2]
        y1,x1,r = blobs[i]
        
        n = 0
        for j,p,c in parameters:
            dx2,dy2 = p[1],p[2]
            y2,x2,r = blobs[j]
            
            if dist((x1+dx1, y1+dy1),(x2+dx2,y2+dy2)) < d**2:
                n+= 1
        neighNum.append(n-1)
    return neighNum
    

def xy_coord(img,offset=(0,0)):
    ''' This function create a list of (x,y) coordinates
    
    :param img: A 2D numpy array
    
    :returns : A numpy 2xN numpy array
    '''
    x0, y0 = offset
    tuple_ = numpy.asarray([(x+x0,y+y0) for y in range(img.shape[0]) for x in range(img.shape[1])])
    return numpy.stack((tuple_[:,0], tuple_[:,1]))
    
    
def xy_value(img):
    ''' This function simply returns a vector of value corresponding to the xy coordinates
    
    :param img: A stack of numpy image
    
    :returns : A numpy 1xN numpy array
    '''
    a,b = img.shape
    return img.reshape(a*b,1)
            

def isin(img, point, radius):
    ''' This function returns True if a point is in a 3D numpy array
    
    :param img: A 2D image
    :param point: The point to be tested
    :param radius: A radius in which the point must be contained from the edges
    
    :returns : A bool
    '''
    x,y = point
    if (x > radius and x < img.shape[0]-radius) and (y > radius and y < img.shape[1]-radius):
        return True
    else:
        return False
        
        
def data_selection(img, blobs, radius, bins=5):
    ''' This function returns the data in [-radius, +radius] from a point in
    both x and y direction
    
    :param img: A 2D image
    :param blob: A list of blob object (y,x,r)
    :param radius: The radius of the selection
    :param bins: Average of the selection
    
    :returns : X and Y data corresponding
    '''
    selection = []
    for i, blob in enumerate(blobs):
        y,x,r = blob
        if isin(img,(x,y),radius):
            X = numpy.mean(img[int(x-radius):int(x+radius), int(y-r):int(y+r)], axis=1)
            Y = numpy.mean(img[int(x-r):int(x+r), int(y-radius):int(y+radius)], axis=0)
            selection.append([i,X,Y])
        else:
            pass
    return selection
    
    
def data2d_selection(img, blobs):
    ''' This function returns the data of size [-r, +r] in both x and
    y coordinate from a point
    
    :param img: A numpy 2D array
    :param blobs: A list of blob object (y,x,r)
    
    :returns : A list of numpy 2D arrays
    '''
    selections = []
    for i,blob in enumerate(blobs):
        y,x,r = blob
        if isin(img,(x,y),r):
            newr = r*1.5
            selections.append([i, img[int(y-newr):int(y+newr), int(x-newr):int(x+newr)]])
        else:
            pass
    return selections
        

def gauss1d(x, amp, mu, sigma):
    ''' This function computes the gaussian function
    
    :param x: The x value to evalute the function to
    :param amp: The amplitude of the function
    :param mu: The average of the function
    :param sigma: The standard deviation of the function
    
    :returns : The evaluated function at the given points
    '''
    inner = (x - mu)**2/(2*sigma**2)
    return amp * numpy.exp(-inner)
    

def gauss2d(xy, amp, x0, y0, sigx, sigy, theta):
    ''' This function computes the 2D gaussian function
    
    :param xy: Tuple of (x,y) coordinates
    :param amp: The amplitude of the function
    :param x0: Center of the peak
    :param y0: Center of the peak
    :param sigx: The standard deviation in x
    :param sigy: The standard deviation in y
    :param theta: Angle of rotation clock-wise
    
    :returns : The evaluated function at the given point
    '''
    x, y = xy
    a = 0.5*((numpy.cos(theta)/sigx)**2 + (numpy.sin(theta)/sigy)**2)
    b = 0.25*(- numpy.sin(2*theta)/(sigx)**2 + numpy.sin(2*theta)/(sigy)**2)
    c = 0.5*((numpy.sin(theta)/sigx)**2 + (numpy.cos(theta)/sigy)**2)
    inner = a*(x-x0)**2
    inner += 2*b*(x-x0)*(y-y0)
    inner += c*(y-y0)**2
    return amp * numpy.exp(-inner)
    

def gaussian_fit(func, selections):
    ''' This function computes the fit of the gaussian function
    It will create its own x and y vectors, since it is not necessary for the current function
    
    :param func: The gaussian function to fit to the data
    :param selections: A list of selections in x and y
    
    :returns : The parameters of the fit for every selections in both x and y
    '''
    parameters = []
    for i, sel in selections:
        xsel, ysel = sel
        x = numpy.arange(0, xsel.shape[0])
        y = numpy.arange(0, ysel.shape[0])
        
        guessX = [numpy.amax(xsel), numpy.argmax(xsel), 10]
        guessY = [numpy.amax(ysel), numpy.argmax(ysel), 10]
        
        try:
            predParamsX, uncertCovX = optimize.curve_fit(func, x, xsel, p0=guessX)
            predParamsY, uncertCovY = optimize.curve_fit(func, y, ysel, p0=guessY)
            parameters.append([i, predParamsX,predParamsY,uncertCovX,uncertCovY])
        except RuntimeError:
            pass
    return parameters
    
    
def gaussian2d_fit(func, selections, blobs):
    ''' This functioncompute the fit of the 2d gaussian function. It will
    create its own x and y vectors
    
    :param func: The 2d gaussian function to fit to the data
    :param selections: A list of numpy 2D arrays
    :param blobs: A list of blobs
    
    :returns : The parameters of the fit for every selections
    '''
    parameters = []
    for i, sel in selections:
        try:
            xy = xy_coord(sel)
            x,y = xy
            index = sel.argmax()
            guess = [numpy.amax(sel), x[index], y[index], 1, 1, 0]
        
            try:
                predParams, uncertCov = optimize.curve_fit(func, xy, numpy.reshape(sel, x.size), p0=guess, 
                    bounds=([0,-numpy.inf,-numpy.inf,0,0,0],[numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,2*numpy.pi]))
                parameters.append([i, predParams, uncertCov])
            except RuntimeError:
                pass
        except IndexError:
            pass
        
    return parameters
    

def tuple2array(l):
    ''' Convert a list of tuple into 2 numpy array
    
    :param l: A list of tuples (x,y)
    
    :returns : A numpy array
    '''
    xvec, yvec = [],[]
    for tup in l:
        x,y = tup
        xvec.append(x)
        yvec.append(y)
    return numpy.asarray([xvec, yvec])
    
    
def sort_var(parameters):
    ''' Sorts the long and short axis of the 2d gaussian fit
    
    :param parameters: The parameters of the selections
    
    :returns : A short and long axis lists
    '''
    s, l = [],[]
    for param in parameters:
        i,p,c = param
        if p[3] < p[4]:
            s.append(p[3])
            l.append(p[4])
        else:
            s.append(p[4])
            l.append(p[3])
    return s, l
    
    
def ratio_var(sAxis,lAxis):
    ''' Computes the ratio between the long and short axis of a gaussian fit
    
    :param sAxis: A list short axis 
    :param lAxis: A list long axis
    
    :returns : A list of ratios
    '''
    return [l/s for s,l in zip(sAxis,lAxis)]
    
    
def intensity(selections,parameters):
    ''' Computes the mean intensity of the cluster and returns the max intensity
    NOTE. An other to return the max intensity would be to return the amp of the gaussian fit
    
    :param selections: A list of numpy 2D array
    :param parameters: A list of list of parameters. Only for the isIDin function
    
    :returns : A list of max intensity and a list on mean intensity of the selection
    '''
    maxInt, meanInt = [],[]
    for i, sel in selections:
        if isIDin(i, parameters):
            maxInt.append(sel.max())
            meanInt.append(sel.mean())
    return maxInt, meanInt
    
    
def isIDin(ID, parameters):
    ''' Finds if the ID is in the list
    
    :param ID: The ID of the list
    :param parmameters: A list of list
    
    :returns: A boolean statement
    '''
    for i,p,c in parameters:
        if ID is i:
            return True
    
    
def area(parameters):
    ''' Computes the area of the gaussian function. The area is approximated to the
    shape of an ellipse. The width is the FWHM of the gaussian fit.
    
    :param parameters: A list of paramters from the fit function
    
    :returns : A list of areas of the gaussian function
    '''
    return [ 2*numpy.log(2)*p[3]*p[4]*numpy.pi for i,p,c in parameters ]
    
    
def float2binary(vec,threshold):
    ''' Converts a float array to binary with a given threshold 
    
    :param vec: A 1D numpy array or a list
    :param threshold: All values over will be set to 1
    
    :param returns: A binary array
    '''
    return numpy.asarray([1 if val >= threshold else 0 for val in vec])
    
    
def get_param(ID, parameters):
    ''' Retreives an array from a list containing the good ID
    
    :param ID: The ID of the array
    :param parameters: A list of parameters, the first element of the list is the ID
    
    :returns : The array
    '''
    for param in parameters:
        if ID is param[0]:
            return param
    
    
def overlap(parameters1, parameters2, blobs, distance):
    ''' Computes the percentage of overlap with their nearest neighbor
    
    :param parameters: Parameters of the gaussian fit
    :param selections: A list of numpy 2D array
    :param distance: A list of distance of nearest neigbor and the index of this neighbor
    
    :returns : A list of percentage of overlap for every parmaters
    '''
    overlap = []
    if parameters1 and parameters2:
        for i,p,c in parameters1:
            w = 100
            threshold = 1e-3
        
            d, nNeigh = distance[i] # ID of nearest neighbor
            if d is -1:
                overlap.append(-1)
            else:
                y1,x1,r1 = blobs[0][i]
                y2,x2,r2 = blobs[1][nNeigh]

                dx,dy = x2-x1, y2-y1

                img = numpy.zeros((w,w))
                xy = xy_coord(img,offset=(-w/2,-w/2))
                gauss1 = float2binary(gauss2d(xy,*p), threshold)
                xy = xy_coord(img,offset=(-(w/2+dx),-(w/2+dy)))
                gauss2 = float2binary(gauss2d(xy, *get_param(nNeigh,parameters2)[1]),threshold)

                overlap.append(sum(gauss1*gauss2)/sum(gauss1))
    
        return overlap
        
    else:
        return [-1]*len(parameters1)


def sub_sample(parameters, blobs, distance=None):
    ''' Sub sample the list of blobs with the parameters that were fitted
    
    :param parameters: A list of parameters
    :param blob: A list of blob parameters
    :parma distance: A list of distance
    
    :returns : A sub_sample list of blobs and a sub sampled list of distance
    '''
    if distance is None:
        return [blobs[i] for i,p,c in parameters if isIDin(i, parameters)]
    else:
        b, d = [], []
        for i, p, c in parameters:
            if isIDin(i, parameters):
                b.append(blobs[i])
                d.append(distance[i])
        return b, d
        
        
def interp_value(y,ymax):
    ''' Interps a y value of an array and returns the index of this value. It's a kind of
    FWHM intercep value.
    
    :param y: A numpy 1D array
    :param ymax: A y value to interp
    
    :returns : The index corresponding where we first meet the half value of ymax
    '''
    for i,elmt in enumerate(y):
        if elmt < (ymax/2):
            break
    return i
    
    
def nearest_value(v,v1,v2):
    ''' Determine the nearest value of a value from two different values
    
    :param v: Value
    :param v1: The first value
    :param v2: The second value
    
    :returns : The nearest value to v
    '''
    if abs(v1-v) < abs(v2-v):
        return v1
    else:
        return v2
        
        
def long_axis_angle(angle, val, l):
    ''' Finds the angle of the long axis
    
    :param angle: The given angle of the parameters
    :param val: The axis length corresponding to the angle
    :param l: The long axis length
    
    :returns : The long axis angle
    '''
    if val is l:
        return angle
    else: 
        return angle + numpy.pi/2
        
    
def find_long_axis_angle(parameters, size=50):
    ''' Finds and returns the angle of the long axis with given parameters
    
    :param parameters: The parameters of the fit
    :param size: The size of the region
    
    :returns : A list of the long axis angle 
    
    NOTE. An angle of -1 should not be considered since the fit is between 0 and 2*pi
    '''
    im = numpy.zeros((size,size))
    lAngle = []
    for i,p,c in parameters:
        xy = xy_coord(im, offset=(-10,-10))
        gauss = gauss2d(xy, *p).reshape(im.shape)
        
        # So that the angle can fit
        dx, dy = numpy.cos((2*numpy.pi) - p[5]), numpy.sin((2*numpy.pi) - p[5])
        try :
            rr, cc = skimage.draw.line(int(p[2]+10), int(p[1]+10), int(p[2]+20*dy+10), int(p[1]+20*dx+10))
            y = gauss[rr,cc]
            x = numpy.arange(y.size)

            ymax = numpy.amax(y)
            index = interp_value(y,ymax)
            val = nearest_value(index,p[3],p[4])
            lAng = long_axis_angle(p[5], val, max(p[3],p[4]))
        except IndexError:
            lAng = -1
        lAngle.append(lAng)
    return lAngle
        

if __name__ == "__main__":
    
    # single particle tracking
    # px = 20 nm

# PLOT THE GAUSSIAN FIT OF THE NEW CLUSTER DETECTION    
    # root = "/Users/Anthony/Documents/Images PDK/Pierre-Luc EXP4/"
    # keepFormat = "*.tif"
    #
    # fileList, nameList = [],[]
    # for path, subdirs, files in os.walk(root):
    #     for name in files:
    #         if fnmatch(name, keepFormat):
    #             fileList.append(os.path.join(path,name))
    #             nameList.append(name)
    #
    # for it,file_ in enumerate(fileList[:5]):
    #     img = norm_img(tifffile.imread(file_))
    #     img = uint2float(img)
    #
    #     blobs = blob_detection(img)
    #
    #     print("Selecting")
    #     selections = data2d_selection(img[0],blobs[0])
    #     print("Fitting the data")
    #     parameters1 = gaussian2d_fit(gauss2d, selections, blobs[0])
    #     print("Fitting is done")
    #     for i,p,c in parameters1:
    #         fig,axes = pyplot.subplots(1,2,figsize=(10,7),tight_layout=True)
    #         axes[0].imshow(selections[i][1])
    #         axes[0].set_title("Selection")
    #
    #         xy = xy_coord(selections[i][1])
    #         z = gauss2d(xy,*p)
    #         axes[1].imshow(z.reshape(selections[i][1].shape))
    #         axes[1].set_title("Gauss2d fit with rotation")
    #
    #         pyplot.savefig("/Users/Anthony/Desktop/gauss2d_fit_rotation/{}_{}.png".format(it, i),dpi=150)
    #         pyplot.close()
    #
    #     print("file : {}/{}".format(it+1, len(fileList[:5])))
        
# FIND THE ANGLE OF THE LONG AXIS
    # img = norm_img(tifffile.imread("test1.tif"))
    # img = uint2float(img)
    #
    # blobs = blob_detection(img)
    # selections = data2d_selection(img[0],blobs[0])
    # parameters = gaussian2d_fit(gauss2d,selections,blobs[0])
    # lAngle = find_long_axis_angle(parameters)
    
# COMPUTE THE MEGAMATRIX
    root = "/Users/Anthony/Documents/Images PDK/Pierre-Luc EXP4/"
    keepFormat = "*.tif"

    fileList, nameList = [],[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, keepFormat):
                fileList.append(os.path.join(path,name))
                nameList.append(name)

    for it,file_ in enumerate(fileList):
        img = norm_img(tifffile.imread(file_))
        img = uint2float(img)

        blobs = blob_detection(img)
        print("Found {} blobs in 640".format(len(blobs[0])))
        print("Found {} blobs in 561".format(len(blobs[1])))
        distances = blob_distance(blobs, 0, 1), blob_distance(blobs, 1, 0)

        selections = data2d_selection(img[0],blobs[0])
        parameters1 = gaussian2d_fit(gauss2d, selections, blobs[0])
        blobs1, distance1 = sub_sample(parameters1, blobs[0], distances[0])
        maxInt1, meanInt1 = intensity(selections,parameters1)
        sVar1, lVar1 = sort_var(parameters1)
        ratioVar1 = ratio_var(sVar1, lVar1)
        area1 = area(parameters1)
        lAngle1 = find_long_axis_angle(parameters1)
        numberOfNeighArea1 = blob_neigh_number(parameters1,blobs[0],12)

        selections = data2d_selection(img[1],blobs[1])
        parameters2 = gaussian2d_fit(gauss2d, selections, blobs[1])
        blobs2, distance2 = sub_sample(parameters2, blobs[1], distances[1])
        maxInt2, meanInt2 = intensity(selections,parameters2)
        sVar2, lVar2 = sort_var(parameters2)
        ratioVar2 = ratio_var(sVar2, lVar2)
        area2 = area(parameters2)
        lAngle2 = find_long_axis_angle(parameters2)
        numberOfNeighArea2 = blob_neigh_number(parameters2,blobs[1],12)

        # overlaps = overlap(parameters1, parameters2, blobs, distance)

        megaArray1 = []
        for i in range(len(parameters1)):
            megaArray1.append(
                [nameList[it],
                parameters1[i][0],
                *list(parameters1[i][1]),
                *blobs1[i],
                *distance1[i],
                maxInt1[i],
                meanInt1[i],
                ratioVar1[i],
                area1[i],
                lAngle1[i],
                numberOfNeighArea1[i]]
            )

        megaArray2 = []
        for i in range(len(parameters2)):
            megaArray2.append(
                [nameList[it],
                parameters2[i][0],
                *list(parameters2[i][1]),
                *blobs2[i],
                *distance2[i],
                maxInt2[i],
                meanInt2[i],
                ratioVar2[i],
                area2[i],
                lAngle2[i],
                numberOfNeighArea2[i]]
            )

        filename = "/Users/Anthony/Desktop/parameters_STED640.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, "a") as csvfile:
            headers = ["ID","amp","x0","y0","sigx","sigy","theta",
                       "y","x","r",
                       "max int","mean int","ratio var","area","lAngle","NumberNeighArea"]
            writer = csv.writer(csvfile,delimiter="\t")
            # if not file_exists:
            #         writer.writerow(headers)
            for rows in megaArray1:
                writer.writerow(rows)

        filename = "/Users/Anthony/Desktop/parameters_STED561.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, "a") as csvfile:
            headers = ["ID","amp","x0","y0","sigx","sigy","theta",
                       "y","x","r",
                       "nNeigh","ID nNeigh",
                       "max int","mean int","ratio var","area","lAngle","NumberNeughArea"]
            writer = csv.writer(csvfile,delimiter="\t")
            # if not file_exists:
            #         writer.writerow(headers)
            for rows in megaArray2:
                writer.writerow(rows)

        print("file : {}/{}".format(it+1, len(fileList)))
    
    
# PLOT THE GAUSSIAN FIT
    # img = norm_img(tifffile.imread("test1.tif"))
    # img = uint2float(img)
    #
    # blobs = blob_detection(img)
    # selections = data2d_selection(img[1],blobs[1])
    # parameters = gaussian2d_fit(gauss2d,selections)
    #
    # for param in parameters:
    #     i,p,c = param
    #     fig,axes = pyplot.subplots(1,2,figsize=(10,7),tight_layout=True)
    #     axes[0].imshow(selections[i])
    #     axes[0].set_title("Selection")
    #
    #     xy = xy_coord(selections[i])
    #     z = gauss2d(xy,*p)
    #     axes[1].imshow(z.reshape(selections[i].shape))
    #     axes[1].set_title("Gauss2d fit with rotation")
    #
    #     pyplot.savefig("/Users/Anthony/Desktop/gauss2d_fit_rotation/{}.png".format(i),dpi=150)
    #     pyplot.close()
        
        
# VARIANCE FOR GAUSS 2D
    # sVars,lVars = [],[]
    # for i,file_ in enumerate(fileList):
    #
    #     img  = norm_img(tifffile.imread(file_))
    #     img = uint2float(img)
    #
    #     blobs = blob_detection(img)
    #
    #     selections = data2d_selection(img[1], blobs[1])
    #     parameters = gaussian2d_fit(gauss2d, selections)
    #
    #     sVar,lVar = sort_var(parameters)
    #
    #     sVars.extend(sVar)
    #     lVars.extend(lVar)
    #
    #     print("file : {}/{}".format(i+1, len(fileList)))
    #
    # with open("/Users/Anthony/Documents/Images PDK/Pierre-Luc EXP3/STED561_gauss2d_rotation_VARIANCE.csv","w") as csvfile:
    #     writer = csv.writer(csvfile,delimiter="\t")
    #     writer.writerow(sVars)
    #     writer.writerow(lVars)
    #
    # fig,axes = pyplot.subplots(2,1,figsize=(6,5),tight_layout=True)
    # axes[0].hist([v for l in varsX for v in l],bins=30,range=(0,60))
    # axes[0].set_title("STD_x")
    # axes[1].hist([v for l in varsY for v in l],bins=30,range=(0,60))
    # axes[1].set_title("STD_y")
    # pyplot.show()
   
    
# VARIANCE FOR GAUSS1D
    # variancesX, variancesY = [],[]
      # for i, file_ in enumerate(fileList):
#         img = norm_img(tifffile.imread(file_))
#         img = uint2float(img)
#
#         threshold = numpy.percentile(img, 98)
#         blobs = blob_detection(img)
#
#         # fig,axes = pyplot.subplots(1,2)
#         # axes[0].imshow(img[0])
#         # axes[1].imshow(img[1])
#         # for blob in blobs[0]:
#         #     y,x,r = blob
#         #     c = pyplot.Circle((x,y), r, fill=False)
#         #     axes[0].add_patch(c)
#         # for blob in blobs[1]:
#         #     y,x,r = blob
#         #     c = pyplot.Circle((x,y), r, fill=False)
#         #     axes[1].add_patch(c)
#
#         distance = blob_distance(blobs)
#
#         selections = data_selection(img[0], blobs[0], 25)
#         parameters = gaussian_fit(gauss1d, selections)
#
#         varianceX = [abs(param[1][2]) for param in parameters]
#         varianceY = [abs(param[2][2]) for param in parameters]
#
#         variancesX.append(varianceX)
#         variancesY.append(varianceY)
#         # for param in parameters:
#         #     i = param[0]
#         #     fig,axes = pyplot.subplots(2,1)
#         #     x = numpy.arange(0,len(selections[i][0]))
#         #     axes[0].plot(x, gauss1d(x,*param[1]))
#         #     axes[0].plot(x, selections[i][0])
#         #     axes[0].set_title("perr = {}".format(numpy.sqrt(numpy.diag(param[3]))))
#         #
#         #     axes[1].plot(x, gauss1d(x,*param[2]))
#         #     axes[1].plot(x, selections[i][1])
#         #     axes[1].set_title("perr = {}".format(numpy.sqrt(numpy.diag(param[4]))))
#
#         # fig,axes = pyplot.subplots(2,1)
#         # axes[0].set_title("IMAGE STED 561")
#         # axes[0].hist(varianceX,bins=30,range=(0,60))
#         # axes[1].hist(varianceY,bins=30,range=(0,60))
#         # pyplot.show()
#
#         print("file : {}/{}".format(i, len(fileList)))
#
#     fig, axes = pyplot.subplots(2,1)
#     axes[0].hist([var for variance in variancesX for var in variance],bins=30, range=(0,60))
#     axes[1].hist([var for variance in variancesY for var in variance],bins=30, range=(0,60))
#     pyplot.show()