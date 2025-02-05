# Python script for linear regression 

import numpy as np
import matplotlib.pyplot as plt

def get_coef(x, y):
    
    # Number of data points
    n = np.size(x)

    # Means of x and y 
    m_x = np.mean(x)
    m_y = np.mean(y)

    # Cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # Regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    
    return (b_0, b_1)
    
def plot_line(x, y, b_0, b_1):
    
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m", marker = "o", s = 30)

    # predicted response vector
    y_pred = b_0 + b_1*x

    # plotting the regression line
    plt.plot(x, y_pred, color = "g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()
    
# Feature vector
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Response vector
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

# estimating coefficients
(b_0, b_1) = get_coef(x, y)
print("Coefficients:\nb_0 = {} \nb_1 = {}".format(b_0, b_1))
    
plot_line(x, y, b_0, b_1)

exit()