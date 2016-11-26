import random
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

train = pandas.read_csv('movie.csv')
train = train.dropna()
y, X = train['imdb_score'], train[['movie_facebook_likes','gross','budget','num_critic_for_reviews','director_facebook_likes','actor_1_facebook_likes',
				'num_voted_users','duration','actor_3_facebook_likes','num_voted_users','cast_total_facebook_likes',
				'num_user_for_reviews','title_year','aspect_ratio','facenumber_in_poster']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# clf = LogisticRegression()
# clf = svm.SVR()
# clf = linear_model.Ridge(alpha=0.5)
# clf = linear_model.Lasso(alpha=1e-5)
# clf = linear_model.ElasticNet(alpha=0.01,l1_ratio=0.5)
clf = tree.DecisionTreeRegressor()
# clf = MLPRegressor (solver='adam', alpha=10,hidden_layer_sizes=(6,2), random_state=3)
clf.fit(X_train, y_train)                         
print(clf.score(X_test,y_test))
a = 0
# for x in xrange(0,len(X_test)):
for x in xrange(0,10):
	# a = y_test[x:x+1].values[0]-clf.predict(X_test[x:x+1])[0]
	# a = abs(a)
	print(y_test[x:x+1].values[0],clf.predict(X_test[x:x+1])[0],y_test[x:x+1].values[0]-clf.predict(X_test[x:x+1])[0])
# print(a/len(X_test))

# print(clf.predict(X_test[0:10]))
# print(y_test[0:10])
# print(X[0:20])
