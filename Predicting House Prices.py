
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation # import animation support

#Generate some houses sizes between 1000 and 3500 (sq meter of house)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low = 85, high = 300, size = num_house )
house_balcony_size=np.random.randint(low=0, high=20, size = num_house )
house_garden_size=np.random.randint(low =0,high = 100,size = num_house)

#Generate house prices from house size with a random noise added
np.random.seed(42)
house_price = house_size * 10000.0 + house_balcony_size*7500.0+house_garden_size*2000.0 + np.random.randint(low = 20000, high = 70000, size = num_house)

#plot generated house and size
plt.plot(house_size, house_price, "bx") #bx = blue x
plt.ylabel("Price")
plt.xlabel("House Size")
plt.show()

plt.plot(house_balcony_size, house_price, "bx") #bx = blue x
plt.ylabel("Price")
plt.xlabel("Balcony Size")
plt.axis([0, 21, 1e6, 3.01e6])
plt.show()

plt.plot(house_garden_size, house_price, "bx") #bx = blue x
plt.ylabel("Price")
plt.xlabel("Garden Size")
plt.axis([0, 101, None, None])
plt.show()

#you need to normalize values to prevent under/overflows - minimize error
def normalize(array):
    return (array - array.mean()) / array.std() #array std is the standard deviation across that array

#define number of training samples, 0.7 = 70%. We can take the first 70% since the values are randomized
num_train_samples = math.floor(num_house * 0.7)  

#define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_balcony_size = np.asarray(house_balcony_size[:num_train_samples])
train_garden_size = np.asarray(house_garden_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_balcony_size_norm = normalize(train_balcony_size)
train_garden_size_norm = normalize(train_garden_size)
train_price_norm = normalize(train_price)

#define test data
test_house_size = np.array(house_size[num_train_samples:])
test_balcony_size = np.array(house_balcony_size[num_train_samples:])
test_garden_size = np.array(house_garden_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])
    
test_house_size_norm = normalize(test_house_size)
test_balcony_size_norm = normalize(train_balcony_size)
test_garden_size_norm = normalize(train_garden_size)
test_house_price_norm = normalize(test_house_price)

#set up the tensorflow placeholders that get updates as we descend down the gradient
tf_house_size = tf.placeholder("float", name="house_size")
tf_balcony_size = tf.placeholder("float", name="balcony_size")
tf_garden_size = tf.placeholder("float", name="garden_size")
tf_price = tf.placeholder("float", name="price")

#Define the variables holding the size_factor and price we set during training
#We initialize them to some random values based on the normal distribution
tf_house_size_factor = tf.Variable(np.random.randn(), name="house_size_factor")
tf_garden_size_factor = tf.Variable(np.random.randn(), name="garden_size_factor")
tf_balcony_size_factor = tf.Variable(np.random.randn(), name="balcony_size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

#2. Define the operations for the predicting values
# Notice, the use of tensorflow and add multiply
# AND the tensorflow mthods understand how to deal
# methods
tf_price_pred = tf.add(tf.add(tf.multiply(tf_house_size_factor, tf_house_size),tf.multiply(tf_balcony_size_factor, tf_balcony_size)),tf.add(tf.multiply(tf_garden_size_factor, tf_garden_size), tf_price_offset))

# 3. Define the Loss function (how much error) - Mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price,2))/(2*num_train_samples)

#Optimizer learning rate. The size of the steps down the gradient
learning_rate = 0.1


# 4. define a Gradient descent optimizer that will minimize the loss defined in the operation "cost".
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Initializing the variables
init = tf.global_variables_initializer()

cost_plot=[]
#Launch the graph in the session
with tf.Session() as sess:
    sess.run(init)
    
    #set how often to display training progres and number of training iterations
    display_every = 2
    num_training_iter = 50
     # calculate the number of lines to animation
    fit_num_plots = math.floor(num_training_iter/display_every)
    # add storage of factor and offset values from each epoch
    fit_house_size_factor = np.zeros(fit_num_plots)
    fit_balcony_size_factor = np.zeros(fit_num_plots)
    fit_garden_size_factor = np.zeros(fit_num_plots)
    fit_price_offsets = np.zeros(fit_num_plots)
    fit_plot_idx = 0    
    
    #keep iterating the training data
    for iteration in range(num_training_iter):
        
        #fit all training data
        for (x1,x2,x3, y) in zip(train_house_size_norm,train_balcony_size_norm, train_garden_size_norm, train_price_norm): #zip aligns both together
            sess.run(optimizer, feed_dict={tf_house_size: x1,tf_balcony_size: x2, tf_garden_size:x3, tf_price: y})
        
        #Display current status
        if (iteration + 1)% display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size:train_house_size_norm,tf_balcony_size:train_balcony_size_norm,tf_garden_size:train_garden_size_norm, tf_price:train_price_norm})
             print("iteration #:", '%04d' % (iteration + 1), "cost=" , "{:.9f}".format(c), "house_size_factor=", sess.run(tf_size_factor),"balcony_size_factor=",sess.run(tf_balcony_size_factor),"graden_size_factor=",sess.run(tf_garden_size_factor), "price_offset=", sess.run(tf_price_offset))
            cost_plot.append(c)
    print("Optimization Finished!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size:train_house_size_norm,tf_balcony_size:train_balcony_size_norm,tf_garden_size:train_garden_size_norm, tf_price:train_price_norm})
    
    print("Trained cost = ", training_cost, "size_factor", sess.run(tf_size_factor), "price_offset = ", sess.run(tf_price_offset), '\n')
    iterr=list(range(25))
    #plotcost function graph for each iteration
    plt.plot(iterr, cost_plot, "bx") #bx = blue x
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    plt.show()







