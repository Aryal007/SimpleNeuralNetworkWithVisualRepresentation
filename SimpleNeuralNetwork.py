import numpy as np  
import matplotlib.pyplot as plt 

# The following is a function definition of the sigmoid function, which is the type of non-linearity chosen for this neural net. 
# It is not the only type of non-linearity that can be chosen, but is has nice analytical features and is easy to teach with. 
# In practice, large-scale deep learning systems use piecewise-linear functions because they are much less expensive to evaluate.  
# The implementation of this function does double duty. If the deriv=True flag is passed in, the function instead calculates the 
# derivative of the function, which is used in the error backpropogation step. 

def sigmoid(x, deriv=False):  
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))  

# The following code creates the input matrix. Although not mentioned in the video, the third column is for accommodating the bias 
# term and is not part of the input. 

X = np.array([[0,0,1], 
            [0,1,1],
            [1,0,1],
            [1,1,1]])

# The output of the exclusive OR function follows. 

y = np.array([[0],
             [1],
             [1],
             [0]])

# The seed for the random generator is set so that it will return the same random numbers each time, 
# which is sometimes useful for debugging.

np.random.seed(1)

syn0 = 8*np.random.random((3,4)) - 4  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 8*np.random.random((4,1)) - 4  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.

# We initialize various arrays to store weights and error at each step to plot it later using matplotlib

numpy_x = []
numpy_y = []

weight1_layer1 = [] 
weight2_layer1 = [] 
weight3_layer1 = [] 
weight4_layer1 = [] 
weight5_layer1 = [] 
weight6_layer1 = [] 
weight7_layer1 = [] 
weight8_layer1 = [] 
weight9_layer1 = [] 
weight10_layer1 = [] 
weight11_layer1 = [] 
weight12_layer1 = [] 
weight1_layer2 = []
weight2_layer2 = []
weight3_layer2 = []
weight4_layer2 = []

# This is the main training loop. The output shows the evolution of the error between the model and desired. 
# The error steadily decreases. 

for j in xrange(6001):  
    
    # Calculate forward through the network.
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    weight1_layer1.append(syn0[0][0])
    weight2_layer1.append(syn0[1][0])
    weight3_layer1.append(syn0[2][0])
    weight4_layer1.append(syn0[0][1])
    weight5_layer1.append(syn0[1][1])
    weight6_layer1.append(syn0[2][1])
    weight7_layer1.append(syn0[0][2])
    weight8_layer1.append(syn0[1][2])
    weight9_layer1.append(syn0[2][2])
    weight10_layer1.append(syn0[0][3])
    weight11_layer1.append(syn0[1][3])
    weight12_layer1.append(syn0[2][3])


    weight1_layer2.append(syn1[0][0])
    weight2_layer2.append(syn1[1][0])
    weight3_layer2.append(syn1[2][0])
    weight4_layer2.append(syn1[3][0])


    # Back propagation of errors using the chain rule. 
    l2_error = y - l2
        print "Error: " + str(np.mean(np.abs(l2_error)))

    numpy_x.append(j)
    numpy_y.append(np.mean(np.abs(l2_error)))

    l2_delta = l2_error*sigmoid(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error * sigmoid(l1,deriv=True)
    
    #update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print "Output after training"
print l2
 
plt.subplot(1,1,1)
plt.plot(numpy_x,numpy_y)
plt.title("Error")
plt.ylabel('Error')
plt.xlabel('Iteration')

plt.show()

plt.subplot(4,3,1)
plt.title('Weights on first layer')
plt.plot(numpy_x,weight1_layer1)
plt.ylabel('Weight 1 ')
plt.xlabel('Iterations')

plt.subplot(4,3,2)
plt.plot(numpy_x,weight2_layer1)
plt.ylabel('Weight 2 ')
plt.xlabel('Iterations')

plt.subplot(4,3,3)
plt.plot(numpy_x,weight3_layer1)
plt.ylabel('Weight 3 ')
plt.xlabel('Iterations')

plt.subplot(4,3,4)
plt.plot(numpy_x,weight4_layer1)
plt.ylabel('Weight 4 ')
plt.xlabel('Iterations')

plt.subplot(4,3,5)
plt.plot(numpy_x,weight5_layer1)
plt.ylabel('Weight 5 ')
plt.xlabel('Iterations')

plt.subplot(4,3,6)
plt.plot(numpy_x,weight6_layer1)
plt.ylabel('Weight 6 ')
plt.xlabel('Iterations')

plt.subplot(4,3,7)
plt.plot(numpy_x,weight7_layer1)
plt.ylabel('Weight 7 ')
plt.xlabel('Iterations')

plt.subplot(4,3,8)
plt.plot(numpy_x,weight8_layer1)
plt.ylabel('Weight 8 ')
plt.xlabel('Iterations')

plt.subplot(4,3,9)
plt.plot(numpy_x,weight9_layer1)
plt.ylabel('Weight 9 ')
plt.xlabel('Iterations')

plt.subplot(4,3,10)
plt.plot(numpy_x,weight10_layer1)
plt.ylabel('Weight 10 ')
plt.xlabel('Iterations')

plt.subplot(4,3,11)
plt.plot(numpy_x,weight11_layer1)
plt.ylabel('Weight 11 ')
plt.xlabel('Iterations')

plt.subplot(4,3,12)
plt.plot(numpy_x,weight12_layer1)
plt.ylabel('Weight 12 ')
plt.xlabel('Iterations')
plt.show()

plt.subplot(4,1,1)
plt.title('Weights on second layer')
plt.plot(numpy_x,weight1_layer2)
plt.ylabel('Weight1')
plt.xlabel('Iterations')

plt.subplot(4,1,2)
plt.plot(numpy_x,weight2_layer2)
plt.ylabel('Weight2')
plt.xlabel('Iterations')

plt.subplot(4,1,3)
plt.plot(numpy_x,weight3_layer2)
plt.ylabel('Weight3')
plt.xlabel('Iterations')

plt.subplot(4,1,4)
plt.plot(numpy_x,weight4_layer2)
plt.ylabel('Weight4')
plt.xlabel('Iterations')

plt.show()



