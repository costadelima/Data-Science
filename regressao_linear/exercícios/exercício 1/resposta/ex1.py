# pip3 install numpy
import numpy as np #biblioteca de mexer com numeros, tipo matriz, tirar media, etc
import pandas as pd #bibli. dataframe
#pip3 install pandas
import matplotlib.pyplot as plt #scatter plot, grafica de dispersao



#WARMUPEXERCISE Example function in octave
#   A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix
def warmUpExercise(n):
	'''
	A = np.array([[1, 2, 3], [3, 4, 5]])
	print(A)
	'''
	#A = np.zeros((5,5))
	A = np.identity(n)
	return A
		

#PLOTDATA Plots the data points x and y into a new figure 
#   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
#   population and profit.
def plotData(X, y):
	plt.scatter(X,y,label='Training data',color="red",marker="x")
	plt.ylabel('Profit in $10,000s') #% Set the y−axis label
	plt.xlabel('Population of City in 10,000s') #% Set the x−axis label
	plt.show()


def plotDataFit(X, y, theta):
	X[2] = theta[0] + theta[1]*X[0]
	plt.scatter(X[0],y,label='Training data',color="red",marker="x")
	plt.ylabel('Profit in $10,000s')
	plt.xlabel('Population of City in 10,000s')
	plt.plot(X[0], X[2],label='Linear regression',color="blue")
	plt.legend(loc='lower right')
	plt.show()



#%COMPUTECOST Compute cost for linear regression
#%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#%   parameter for linear regression to fit the data points in X and y
def computeCost(X, y, theta):
	a = 1/(2*m)
	somatorio = 0 
	for i in range(0, m):
		somatorio = somatorio + ((theta[0] + theta[1]*X.iloc[i][0])-(y.iloc[i]))**2
	return a*somatorio

#%GRADIENTDESCENT Performs gradient descent to learn theta
#%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
#%   taking num_iters gradient steps with learning rate alpha
def gradientDescent(X, y, theta, alpha, num_iters):
	#J = dict()
	for iterac in range(0, num_iters):
		#print('theta',theta)
		#print('J',computeCost(X,y,theta))		
		#J[str(theta)] = computeCost(X,y,theta)
		somatorio = 0		
		for i in range(0, m):
			somatorio = somatorio + ((theta[0] + theta[1]*X.iloc[i][0])-(y.iloc[i]))*X.iloc[i][1]		
		newtheta0 = theta[0] - (alpha*(1/m)*somatorio)	
		somatorio = 0
		for i in range(0, m):
			somatorio = somatorio + ((theta[0] + theta[1]*X.iloc[i][0])-(y.iloc[i]))*X.iloc[i][0]		
		newtheta1 = theta[1] - (alpha*(1/m)*somatorio)
		theta[0] = newtheta0
		theta[1] = newtheta1	
	#J[str(theta)] = computeCost(X,y,theta)
	#return J
	return theta
			

# ==================== Part 1: Basic Function ====================
print('Runnig warmUpExercise .. \n')
print('5x5 Identity Matrix: \n')
print(warmUpExercise(5))
#input() #pausa

# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = pd.read_csv('ex1data1.txt', delimiter = ',', header = None) #read comma separated data
X = data[0]
y = data[1]
m = len(y) #number of training examples
plotData(X,y)
#input() 

#%% =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')
X = X.to_frame() #serie para dataFrame
X[1] = 1 # Add a column of ones to x
theta = np.array([0 , 0], dtype = float) #initialize fitting parameters

#Some gradient descent settings
iterations = 1500
alpha = 0.01 #learnig rate

#% compute and display initial cost
J = computeCost(X, y, theta)
print(J) # custo theta[0,0]

#% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)

#% print theta to screen
print('Theta found by gradient descent: ')
print(theta[0],theta[1])

#% Plot the linear fit
plotDataFit(X, y, theta)

#% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] * theta
print('For population = 35,000, we predict a profit of ',np.sum(predict1) * 10000)
predict2 = [1, 7] * theta;
print('For population = 70,000, we predict a profit of ',np.sum(predict2) * 10000)



























