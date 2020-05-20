import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

def loadData():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten/255
    test_set_x = test_set_x_flatten/255
    # num_px = train_set_x_orig.shape[0]
    
    return train_set_x, train_set_y, test_set_x, test_set_y

# Loading the data (cat/non-cat)
def play():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    # Example of a picture
    index = 10
    # plt.imshow(train_set_x_orig[index])
    # print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
    # plt.show()
    
    # print(train_set_x_orig[index].shape)
    # print(train_set_x_orig[index])
    
    '''
    - m_train (number of training examples)
    - m_test (number of test examples)
    - num_px (= height = width of a training image)

    '''
    m_train = train_set_x_orig.shape[0]
    # print('#### train')
    # print(train_set_x_orig.shape) # (209, 64, 64, 3)
    # print(m_train) # 209
    m_test = test_set_x_orig.shape[0]
    # print('>>>> test')
    # print(test_set_x_orig.shape)
    # print(m_test)
    
    num_px = train_set_x_orig.shape[1]
    # print ("Number of training examples: m_train = " + str(m_train))
    # print ("Number of testing examples: m_test = " + str(m_test))
    # print ("Height/Width of each image: num_px = " + str(num_px))
    # print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    # print ("train_set_x shape: " + str(train_set_x_orig.shape))
    # print ("train_set_y shape: " + str(train_set_y.shape))
    # print ("test_set_x shape: " + str(test_set_x_orig.shape))
    # print ("test_set_y shape: " + str(test_set_y.shape))
    
    '''
    Exercise: Reshape the training and test data sets so that images of size (num_px, num_px, 3) are 
    flattened into single vectors of shape (num_px  ∗∗  num_px  ∗∗  3, 1)
    A trick when you want to flatten a matrix X of shape (a,b,c,d) 
    to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a) is to use:
        X_flatten = X.reshape(X.shape[0], -1).T      
    # X.T is the transpose of X
    '''
    ### START CODE HERE ### (≈ 2 lines of code)
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    ### END CODE HERE ###
    # print(train_set_x_orig)
    # print(train_set_x_orig.shape)
    # print('------------------')
    # print(train_set_x_flatten)
    # print(train_set_x_flatten.shape)
    # print(64*64*3)
    
    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
    
    '''
    What you need to remember:
    >> Common steps for pre-processing a new dataset are:
    1> Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
    2> Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
    3> "Standardize" the data
    '''
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.

# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(z * (-1)) )
    ### END CODE HERE ###
    
    return s

def test_sigmoid():
    print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
    
    
'''
4.2 - Initializing parameters¶
Exercise: Implement parameter initialization in the cell below. 
You have to initialize w as a vector of zeros. 
If you don't know what numpy function to use, 
look up np.zeros() in the Numpy library's documentation.
'''
# GRADED FUNCTION: initialize_with_zeros
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros(shape=(dim, 1))
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def test_initialize_with_zeros():
    dim = 2
    w, b = initialize_with_zeros(dim)
    print ("w = " + str(w))
    print ("b = " + str(b))
    
# 4.3 - Forward and Backward propagation
# GRADED FUNCTION: propagate

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    # compute activation
    A = sigmoid(np.dot(w.T, X) + b)                              
    # A = sigmoid(w.T @ X + b) 
    # https://numpy.org/devdocs/reference/generated/numpy.dot.html
    
    # compute cost
    # cost = -1 / (m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))                                
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) 
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = 1 / m * X @ (A - Y).T
    db = 1 / m * np.sum(A - Y)
    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def test_propagate():
    w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
    print('> w')
    print(w)
    print('=========')
    print('> b')
    print(b)
    print('=========')
    print('> X')
    print(X)
    print('=========')
    print('Y')
    print(Y)
    grads, cost = propagate(w, b, X, Y)
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    print ("cost = " + str(cost))


'''
4.4 - Optimization
You have initialized your parameters.
You are also able to compute a cost function and its gradient.
Now, you want to update the parameters using gradient descent.
Exercise: Write down the optimization function. 
The goal is to learn  ww  and  bb  by minimizing the cost function  J . 
For a parameter  θ , 
the update rule is  θ = θ − α * dθ , where  α  is the learning rate.
'''
# GRADED FUNCTION: optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs 

def test_optimize():
    w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
    params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
    print ("w = " + str(params["w"]))
    print ("b = " + str(params["b"]))
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    
    

# GRADED FUNCTION: predict
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)
    print('>A: ')
    print(A)
    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        Y_prediction[:, i] = (A[:, i] > 0.5) * 1
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def test_predict():
    w = np.array([[0.1124579],[0.23106775]])
    b = -0.3
    X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
    print ("predictions = " + str(predict(w, b, X)))

# 5 - Merge all functions into a model¶
'''
You will now see how the overall model is structured by putting together all the building blocks (functions implemented in the previous parts) together, in the right order.

Exercise: Implement the model function. Use the following notation:

- Y_prediction_test for your predictions on the test set
- Y_prediction_train for your predictions on the train set
- w, costs, grads for the outputs of optimize()
'''
# GRADED FUNCTION: model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

def test_model():
    train_set_x, train_set_y, test_set_x, test_set_y = loadData()
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    # draw the cost line out
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()
    
    
# 6 - Further analysis (optional/ungraded exercise)
# Choice of learning rate
# Reminder: In order for Gradient Descent to work you must choose the learning rate wisely. 
# > The learning rate $\alpha$ determines how rapidly we update the parameters. 
# > If the learning rate is too large we may "overshoot" the optimal value. 
# > Similarly, if it is too small we will need too many iterations to converge to the best values. 
# > That's why it is crucial to use a well-tuned learning rate.
def learning_rate():
    train_set_x, train_set_y, test_set_x, test_set_y = loadData()
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
        print ('\n' + "-------------------------------------------------------" + '\n')

        for i in learning_rates:
            plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

            plt.ylabel('cost')
            plt.xlabel('iterations (hundreds)')

            legend = plt.legend(loc='upper center', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            plt.show()

'''
Interpretation:
Different learning rates give different costs and thus different predictions results.
If the learning rate is too large (0.01), the cost may oscillate up and down. It may even diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost).
A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.
In deep learning, we usually recommend that you:
Choose the learning rate that better minimizes the cost function.
If your model overfits, use other techniques to reduce overfitting.
'''

def test_with_own_image():
    train_set_x, train_set_y, test_set_x, test_set_y = loadData()
    
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    ## START CODE HERE ## (PUT YOUR IMAGE NAME) 
    my_image = "my_image.jpg"   # change this to the name of your image file 
    ## END CODE HERE ##

    # We preprocess the image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    image = image/255.
    num_px = image.shape[0]
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    

if __name__ == "__main__":
    # play()
    # test_sigmoid()
    # test_initialize_with_zeros()
    # test_propagate()
    # test_optimize()
    # test_predict()
    test_model()
    # learning_rate()
    # test_with_own_image()








