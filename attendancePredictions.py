import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


WorldCupMatches = pd.read_csv(r'WorldCupMatches.csv')
#print(WorldCupMatches.shape)
#print(WorldCupMatches.describe())

Years = np.array(WorldCupMatches['Year'])
Years = Years[::-1]
print(Years)
GameAttendance = np.array(WorldCupMatches['Game'])
print(GameAttendance)
GameAttendance[::-1]


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









