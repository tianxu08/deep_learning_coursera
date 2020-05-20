import numpy as np
import time
from scipy.misc import imread, imsave, imresize
import imageio
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# http://cs231n.github.io/python-numpy-tutorial/


# python3 -m pip install numpy // install python3 package
def test():
    print('hello world')
    # np.array(1, 2, 3)
    print(np.array([1, 2, 3]))

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    
    return s

def image2vector(image):
    # 
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v

def sigmoid_test():
    x = np.array([1, 2, 3])
    print(sigmoid(x))

def image2vector_test():
    image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

    print ("image2vector(image) = " + str(image2vector(image))) 
    

def vectors():
    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
    dot = np.dot(x1, x2)
    print(">>>> " + str(dot))
    outer = np.outer(x1, x2)
    print(">>>> outer")
    print(outer)
    print("#### multiple")
    mul = np.multiply(x1, x2)
    print(mul)

def math_numpy1():
    x = np.array([[1,2],[3,4]], dtype=np.float64)
    y = np.array([[5,6],[7,8]], dtype=np.float64)

    # Elementwise sum; both produce the array
    # [[ 6.0  8.0]
    #  [10.0 12.0]]
    print(x + y)
    print(np.add(x, y))

    # Elementwise difference; both produce the array
    # [[-4.0 -4.0]
    #  [-4.0 -4.0]]
    print(x - y)
    print(np.subtract(x, y))

    # Elementwise product; both produce the array
    # [[ 5.0 12.0]
    #  [21.0 32.0]]
    print(x * y)
    print(np.multiply(x, y))

    # Elementwise division; both produce the array
    # [[ 0.2         0.33333333]
    #  [ 0.42857143  0.5       ]]
    print(x / y)
    print(np.divide(x, y))

    # Elementwise square root; produces the array
    # [[ 1.          1.41421356]
    #  [ 1.73205081  2.        ]]
    print(np.sqrt(x))

# compute inner products of vectors, 
# to multiply a vector by a matrix, and
# to multiply matrices
def math_numpy2():
    x = np.array([[1,2],[3,4]])
    y = np.array([[5,6],[7,8]])

    v = np.array([9,10])
    w = np.array([11, 12])
    
    print(v.dot(w))
    print(np.dot(v, w))
    print(np.dot(w, v))
    
    print(x.dot(v))
    print(np.dot(x, v))
    
    print(x.dot(y))
    print(np.dot(x,y))

def math_numpy3():
    x = np.array([[1, 2], [3, 4]])
    print(np.sum(x))
    print(np.sum(x, axis=0))
    print(np.sum(x, axis=1))

def math_numpy4():
    x = np.array([[1,2], [3,4], [5, 6]])
    print(x)
    print(x.T)
    
def broadcasting():
    x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
    v = np.array([1, 0, 1])
    y = np.empty_like(x)   #
    
    for i in range(4):
        y[i, :] = x[i, :] + v
    print(x)
    print(y)
    vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
    print(vv)
    y2 = x + vv
    print('>> y2')
    print(y2)
    
    y3 = x + v
    print('>>> y3')
    print(y3)
   
def sciPy():
     img = imread('./husky.jpg')
     print(img.dtype, img.shape)
     img2 = imageio.imread('./husky.jpg')
     print(img2.dtype, img2.shape)
     img_tinted = img2 * [1, 0.95, 0.9]
     img_tinted = imresize(img_tinted, (300, 300))
    #  imsave('./husky_tinted.jpg', img_tinted)
     imageio.imwrite('./husky_tinted2.jpg', img_tinted)
     '''
        note.py:143: DeprecationWarning: `imresize` is deprecated!
        `imresize` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
        Use ``skimage.transform.resize`` instead.
        img_tinted = imresize(img_tinted, (300, 300))
        note.py:144: DeprecationWarning: `imsave` is deprecated!
        `imsave` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
        Use ``imageio.imwrite`` instead.
     '''

def matlab_files():
    # Create the following array where each row is a point in 2D space:
    # [[0 1]
    #  [1 0]
    #  [2 0]]
    x = np.array([[0, 1], [1, 0], [2, 0]])
    print(x)

    # Compute the Euclidean distance between all rows of x.
    # d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
    # and d is the following array:
    # [[ 0.          1.41421356  2.23606798]
    #  [ 1.41421356  0.          1.        ]
    #  [ 2.23606798  1.          0.        ]]
    d = squareform(pdist(x, 'euclidean'))
    print(d)

'''
Matplotlib is a plotting library. 
In this section give a brief introduction to the matplotlib.pyplot module, 
which provides a plotting system similar to that of MATLAB.
'''
def matplotlib():
    # Compute the x and y coordinates for points on a sine curve
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)

    # Plot the points using matplotlib
    plt.plot(x, y)
    plt.show()  # You must call plt.show() to make graphics appear.

def matplotlib2():
    # Compute the x and y coordinates for points on sine and cosine curves
    x = np.arange(0, 3 * np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # Plot the points using matplotlib
    plt.plot(x, y_sin)
    plt.plot(x, y_cos)
    plt.xlabel('x axis label')
    plt.ylabel('y axis label')
    plt.title('Sine and Cosine')
    plt.legend(['Sine', 'Cosine'])
    plt.show()
    
def matplotlib3_Subplots():
    # Compute the x and y coordinates for points on sine and cosine curves
    x = np.arange(0, 3 * np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # Set up a subplot grid that has height 2 and width 1,
    # and set the first such subplot as active.
    plt.subplot(2, 1, 1)

    # Make the first plot
    plt.plot(x, y_sin)
    plt.title('Sine')

    # Set the second subplot as active, and make the second plot.
    plt.subplot(2, 1, 2)
    plt.plot(x, y_cos)
    plt.title('Cosine')

    # Show the figure.
    plt.show()
  
def images():    
    img = imread('./husky.jpg')
    img_tinted = img * [1, 0.95, 0.9]

    # Show the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    # Show the tinted image
    plt.subplot(1, 2, 2)

    # A slight gotcha with imshow is that it might give strange results
    # if presented with data that is not uint8. To work around this, we
    # explicitly cast the image to uint8 before displaying it.
    plt.imshow(np.uint8(img_tinted))
    plt.show()  

if __name__ == "__main__":
    # sigmoid_test()
    # image2vector_test()
    # vectors()
    # math_numpy2()
    # math_numpy3()
    # math_numpy4()
    # broadcasting()
    # sciPy()
    # matlab_files()
    # matplotlib()
    # matplotlib2()
    # matplotlib3_Subplots()
    images()
