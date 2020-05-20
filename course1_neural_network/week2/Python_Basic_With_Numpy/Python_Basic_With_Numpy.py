import math
import numpy as np
import time

# 1 - Building basic functions with numpy
# 1.1 - sigmoid function, np.exp()
def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + math.exp(x * (-1)))
    ### END CODE HERE ###
    
    return s

def test_basic_sigmoid():
    x = 3
    print(basic_sigmoid(x))
    
def test_np_array():
    x = np.array([1, 2, 3])
    print(np.exp(x))
    print(x + 3)

# GRADED FUNCTION: sigmoid
def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1 + np.exp(-x))
    ### END CODE HERE ###
    
    return s
    
def test_sigmoid():
    x = np.array([1, 2, 3])
    print(sigmoid(x))
    
# 1.2 - Sigmoid gradient¶
# GRADED FUNCTION: sigmoid_derivative
def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    s = sigmoid(x)
    ds = s * (1 - s)
    ### END CODE HERE ###
    return ds   
    
def test_sigmoid_derivative():
    x = np.array([1, 2, 3])
    print("x = " + str(x))
    print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
    
# 1.3 - Reshaping arrays
# X.shape() is ued to get the shape(dimension) of a matrix/vector X
# X.reshape(...) is used to reshape X into some other dimension
# GRADED FUNCTION: image2vector
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    ### END CODE HERE ###
    
    return v

def test_image2vector():
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

# 1.4 - Normalizing rows
# GRADED FUNCTION: normalizeRows
# ToDo: learn more about np.linalg.norm
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord = 2, keepdims = True)
    # Divide x by its norm.
    x = x / x_norm
    ### END CODE HERE ###
    return x

def test_normalizeRows():
    x = np.array([[0, 3, 4], [1, 6, 4]])
    print("normalizeRows(x) = " + str(normalizeRows(x)))

# 1.5 - Broadcasting and the softmax function
# Exercise: Implement a softmax function using numpy. 
# You can think of softmax as a normalizing function 
# used when your algorithm needs to classify two or more classes. 
# You will learn more about softmax in the second course of 
# this specialization.
# GRADED FUNCTION: softmax
def softmax(x):
    """Calculates the softmax for each row of the input x.
    Your code should work for a row vector and also for matrices of shape (m,n).
    Argument:
    x -- A numpy matrix of shape (m,n)
    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    
    ### START CODE HERE ### (≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)
    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum
    ### END CODE HERE ###
    return s

def test_softmax():
    x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
    print("softmax(x) = " + str(softmax(x)))
    
# What you need to remember:
# np.exp(x) works for any np.array x and applies the exponential function to every coordinate
# the sigmoid function and its gradient
# image2vector is commonly used in deep learning
# np.reshape is widely used. In the future, you'll see that keeping your matrix/vector dimensions straight will go toward eliminating a lot of bugs.
# numpy has efficient built-in functions
# broadcasting is extremely useful

# 2) Vectorization
def non_vectorization():
    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
    
    ### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
    dot = 0
    for i in range(len(x1)):
        dot += x1[i] * x2[i]
    print(dot)
    
    ### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
    outer = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            outer[i, j] = x1[i] * x2[j]
    print("outer" + str(outer))
    
    ### CLASSIC ELEMENTWISE IMPLEMENTATION ###
    mul = np.zeros(len(x1))
    for i in range(len(x1)):
        mul[i] = x1[i] * x2[i]
    print ("elementwise multiplication = " + str(mul))
    
    ### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
    W = np.random.rand(3, len(x1))
    gdot = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        for j in range(len(x1)):
            gdot[i] += W[i, j] * x1[j]
            
    print("gdot = " + str(gdot))
    
def vectorization():
    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
    ### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
    dot = np.dot(x1, x2)
    print(dot)
    
    ### VECTORIZED OUTER PRODUCT ###
    outer = np.outer(x1, x2)
    print("outer" + str(outer))
    
    ### CLASSIC ELEMENTWISE IMPLEMENTATION ###
    mul = np.multiply(x1, x2)
    print ("elementwise multiplication = " + str(mul))
    
     ### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
    W = np.random.rand(3, len(x1))
    gdot = np.dot(W, x1)            
    print("gdot = " + str(gdot))

# 2.1 Implement the L1 and L2 loss functions
# GRADED FUNCTION: L1
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(np.abs(yhat - y))
    ### END CODE HERE ###
    
    return loss

def test_L1():
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L1 = " + str(L1(yhat,y)))

# 2.2 GRADED FUNCTION: L2
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(np.square(yhat - y))
    ### END CODE HERE ###
    
    return loss

def test_L2():
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L2 = " + str(L2(yhat,y)))

# What to remember:
# Vectorization is very important in deep learning. 
#    It provides computational efficiency and clarity.
# You have reviewed the L1 and L2 loss.
# You are familiar with many numpy functions 
#    such as np.sum, np.dot, np.multiply, np.maximum, etc...

if __name__ == "__main__":
    # test_basic_sigmoid()
    # test_np_array()
    # test_sigmoid()
    # test_sigmoid_derivative()
    # test_image2vector()
    # test_normalizeRows()
    # test_softmax()
    # non_vectorization()
    # vectorization()
    # test_L1()
    #test_L2()



