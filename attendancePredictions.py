import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def PCA():
	df = pd.read_excel(r'SOCCER STADIUM.xlsm', names=['FEDERATION', 'STADIUM NAME', 'CITY', 'CAPACITY', 'COUNTRY', 'Population of the city', 'Revenue of the stadium', 'Avg age ', 'Percentage of Females', 'Percentage Males', 'Year Opened'])
	#print(df)

	features = ['CAPACITY', 'Population of the city','Avg age ', 'Percentage of Females', 'Percentage Males', 'Year Opened']

	x = df.loc[:, features].values
	y = df.loc[:,['Revenue of the stadium']].values
	x = StandardScaler().fit_transform(x)

	pca = PCA(n_components = 3)
	principalComponents = pca.fit_transform(x)
	print(principalComponents.shape)

	ex_variance=np.var(principalComponents,axis=0)
	ex_variance_ratio = ex_variance/np.sum(ex_variance)
	print(ex_variance_ratio)


	plt.matshow(pca.components_,cmap='viridis')
	plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
	plt.colorbar()
	plt.xticks(range(len(features)),features, rotation=65,ha='left')
	plt.tight_layout()
	plt.show()# 

	#principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

	#finalDf = pd.concat([principalDf, df[['Revenue of the stadium']]], axis = 1)

	#print(finalDf)



	def myplot(score,coeff,labels=None):
	    xs = score[:,0]
	    ys = score[:,1]
	    n = coeff.shape[0]
	    scalex = 1.0/(xs.max() - xs.min())
	    scaley = 1.0/(ys.max() - ys.min())
	    plt.scatter(xs * scalex,ys * scaley, c = y)
	    for i in range(n):
	        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
	        if labels is None:
	            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
	        else:
	            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

	plt.xlim(-1,1)
	plt.ylim(-1,1)
	plt.xlabel("PC{}".format(1))
	plt.ylabel("PC{}".format(2))
	plt.grid()

	myplot(principalComponents[:,0:2],np.transpose(pca.components_[0:2, :]))
	plt.show()

	print(pca.explained_variance_ratio_)
	print(abs( pca.components_ ))


	model = PCA(n_components=3).fit(train_features)
	X_pc = model.transform(train_features)

	# number of components
	n_pcs= model.components_.shape[0]


	most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

	initial_feature_names = ['CAPACITY', 'Population of the city','Avg age ', 'Percentage of Females', 'Percentage Males', 'Year Opened']

	# get the names
	most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

	# LIST COMPREHENSION HERE AGAIN
	dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

	# build the dataframe
	df = pd.DataFrame(dic.items())
	print(df)


def classify()




#print(stadiumData)
#print(WorldCupMatches.shape)
#print(WorldCupMatches.describe())

def predict():




	from sklearn import linear_model


	WorldCupMatches = pd.read_csv(r'WorldCupMatches.csv')

	df = pd.DataFrame(WorldCupMatches, columns =['Year', 'Game', 'Average Attendance'])
	print(df)


	X = df[['Year','Average Attendance']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
	Y = df['Game']

	regr = linear_model.LinearRegression()
	regr.fit(X, Y)

	Year = 2026
	AverageAttendance = 4000000
	print ('Predicted Game Attendance: \n', regr.predict([[Year , AverageAttendance]]))

	mpl.rcParams['legend.fontsize'] = 12
	fig = plt.figure() 
	ax = fig.gca(projection ='3d') 

	x = np.array(X)
	y = np.array(Y)

	ax.scatter(x[:, 1], x[:, 2], y, label ='y', s = 5) 
	ax.legend() 
	ax.view_init(45, 0) 
  
	plt.show() 

	Years = np.array(WorldCupMatches['Year'])
	Years = Years[::-1]

	GameAttendance = list(WorldCupMatches['Game'])
	GameAttendance = GameAttendance[::-1]


	i = 0
	for game in GameAttendance:
		print(game)
		#game = game.replace(',', '')
		GameAttendance[i] = game
		i += 1

	GameAttendance = list(map(int, GameAttendance))
	new = np.array([[x] for x in GameAttendance])  

	GameAttendance = list(WorldCupMatches['Game'])



	Years = Years.reshape(-1, 1)
	new.reshape(-1,1)


	X_train, X_test, y_train, y_test = train_test_split(Years, new, test_size=1/5, random_state=0)

	regressor = LinearRegression()
	regressor.fit(X_train, y_train)


	viz_train = plt
	viz_train.scatter(X_train, y_train, color='red')
	viz_train.plot(X_train, regressor.predict(X_train), color='blue')
	viz_train.title('Year vs Attendance (Training set)')
	viz_train.xlabel('Year')
	viz_train.ylabel('Attendance')
	viz_train.show()


	viz_train = plt
	viz_train.scatter(X_train, y_train, color='red')
	viz_train.plot(X_train, regressor.predict(X_train), color='blue')
	viz_train.title('Year vs Attendance (Training set)')
	viz_train.xlabel('Year')
	viz_train.ylabel('Attendance')
	viz_train.show()



	# Visualizing the Test set results
	viz_test = plt
	viz_test.scatter(X_test, y_test, color='red')
	viz_test.plot(X_train, regressor.predict(X_train), color='blue')
	viz_test.title('Year vs Attendance (Training set)')
	viz_test.xlabel('Year')
	viz_test.ylabel('Attendance')
	viz_test.show()



	y_pred = regressor.predict([[2026]])
	print(y_pred)
	#from the csv
	#years = np.array([1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014]).reshape(-1,1)
	averages = []
	#print(WorldCupMatches['Attendance'])

	# _1930 = WorldCupMatches.query('Year==1930')['Attendance']
	# _1934 = WorldCupMatches.query('Year==1934')['Attendance']
	# _1938 = WorldCupMatches.query('Year==1938')['Attendance']
	# _1950 = WorldCupMatches.query('Year==1950')['Attendance']
	# _1954 = WorldCupMatches.query('Year==1954')['Attendance']
	# _1958 = WorldCupMatches.query('Year==1958')['Attendance']
	# _1962 = WorldCupMatches.query('Year==1962')['Attendance']
	# _1966 = WorldCupMatches.query('Year==1966')['Attendance']
	# _1970 = WorldCupMatches.query('Year==1970')['Attendance']
	# _1974 = WorldCupMatches.query('Year==1974')['Attendance']
	# _1978 = WorldCupMatches.query('Year==1978')['Attendance']
	# _1982 = WorldCupMatches.query('Year==1982')['Attendance']
	# _1986 = WorldCupMatches.query('Year==1986')['Attendance']
	# _1990 = WorldCupMatches.query('Year==1990')['Attendance']
	# _1994 = WorldCupMatches.query('Year==1994')['Attendance']
	# _1998 = WorldCupMatches.query('Year==1998')['Attendance']
	# _2002 = WorldCupMatches.query('Year==2002')['Attendance']
	# _2006 = WorldCupMatches.query('Year==2006')['Attendance']
	# _2010 = WorldCupMatches.query('Year==2010')['Attendance']
	# _2014 = WorldCupMatches.query('Year==2014')['Attendance']

	# avg1930 = np.average(_1930)
	# averages.append(avg1930)

	# avg1934 = np.average(_1934)
	# averages.append(avg1934)

	# avg1938 = np.average(_1938)
	# averages.append(avg1938)

	# avg1950 = np.average(_1950)
	# averages.append(avg1950)

	# avg1954 = np.average(_1954)
	# averages.append(avg1954)

	# avg1958 = np.average(_1958)
	# averages.append(avg1958)

	# avg1962 = np.average(_1962)
	# averages.append(avg1962)

	# avg1966 = np.average(_1966)
	# averages.append(avg1966)

	# avg1970 = np.average(_1970)
	# averages.append(avg1970)

	# avg1974 = np.average(_1974)
	# averages.append(avg1974)

	# avg1978 = np.average(_1978)
	# averages.append(avg1978)

	# avg1982 = np.average(_1982)
	# averages.append(avg1982)

	# avg1986 = np.average(_1986)
	# averages.append(avg1986)

	# avg1990 = np.average(_1990)
	# averages.append(avg1990)

	# avg1994 = np.average(_1994)
	# averages.append(avg1994)

	# avg1998 = np.average(_1998)
	# averages.append(avg1998)

	# avg2002 = np.average(_2002)
	# averages.append(avg2002)

	# avg2006 = np.average(_2006)
	# averages.append(avg2006)

	# avg2010 = np.average(_2010)
	# averages.append(avg2006)

	# avg2014 = np.average(_2014)
	# averages.append(avg2014)


	# X_train, X_test, y_train, y_test = train_test_split(years, averages, test_size=0.2, random_state=0)

	# regressor = LinearRegression()  
	# regressor.fit(X_train, y_train) #training the algorithm


	# r_sq = regressor.score(X_train,_train)
	# print('coefficient of det:', r_sq)

	# #To retrieve the intercept:
	# print('intercept: ', regressor.intercept_)
	# #For retrieving the slope:
	# print('slope: ', regressor.coef_)

	# y_pred = regressor.predict(X_test)
	# print('predicted response: ', y_pred)


	# plt.clf()
	# plt.plot(years, averages)
	# plt.title('Attendance over Time')
	# plt.xlabel('Year')
	# plt.ylabel('Attendance')
	# plt.show()

#predict()


def test():
	import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	# create arrays for the data points
	X = []
	Y = []

	#read the csv file
	csvReader = open('WorldCupMatches.csv')

	#skips the header line
	csvReader.readline()

	for line in csvReader:
	    y, x1, x2 = line.split(',')
	    X.append([float(x1), float(x2), 1]) # add the bias term at the end
	    Y.append(float(y))


	print(X)
	print(Y)
	# use numpy arrays so that we can use linear algebra later
	X = np.array(X)
	Y = np.array(Y)

	# graph the data
	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X[:, 0], X[:, 1], Y)
	ax.set_xlabel('Viewership')
	ax.set_ylabel('Year')
	ax.set_zlabel('Avg. Game Attendance')

	# Use Linear Algebra to solve
	a = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
	predictedY = np.dot(X, a)

	# calculate the r-squared
	SSres = Y - predictedY
	SStot = Y - Y.mean()
	rSquared = 1 - (SSres.dot(SSres) / SStot.dot(SStot))
	print("the r-squared is: ", rSquared)
	print("the coefficient (value of a) for age, weight, constant is: ", a)

	# create a wiremesh for the plane that the predicted values will lie
	xx, yy, zz = np.meshgrid(X[:, 0], X[:, 1], X[:, 2])
	combinedArrays = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
	Z = combinedArrays.dot(a)

	# graph the original data, predicted data, and wiremesh plane
	fig = plt.figure(2)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X[:, 0], X[:, 1], Y, color='r', label='Actual Game Attendance')
	ax.scatter(X[:, 0], X[:, 1], predictedY, color='g', label='Predicted Game Attendance')
	ax.plot_trisurf(combinedArrays[:, 0], combinedArrays[:, 1], Z, alpha=0.5)
	ax.set_xlabel('Viewership')
	ax.set_ylabel('Year')
	ax.set_zlabel('Game Attendance')
	ax.legend()
	plt.show()

test();




