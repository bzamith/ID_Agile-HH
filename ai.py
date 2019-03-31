import pandas as pd #For datasets
import os #Create dir
from sklearn.neighbors import KNeighborsClassifier #knn
from sklearn import svm #svm
from sklearn.neural_network import MLPClassifier #mlp
from sklearn.naive_bayes import GaussianNB #nb
from sklearn import tree #dtree
from sklearn.ensemble import RandomForestClassifier #rforest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans

def initialize(data):
	count_vect = CountVectorizer(stop_words='english')
	tf_transformer = TfidfTransformer()
	data_train, data_test = train_test_split(data,test_size=0.2,stratify=data["especialist"])
	return count_vect,tf_transformer,data_train,data_test

def tfidf(data, count_vect, tf_transformer, mode="train"):
	if mode.upper() == "TRAIN":
		attr = count_vect.fit_transform(data["phrase"].values)
		attr = tf_transformer.fit_transform(attr)
	elif mode.upper() == "TEST":
		attr = count_vect.transform(data["phrase"].values)
		attr = tf_transformer.transform(attr)
	return attr

def nlp(data):
	count_vect,tf_transformer,data_train,data_test = initialize(data)
	attr_train = tfidf(data_train, count_vect, tf_transformer)
	attr_test = tfidf(data_test, count_vect, tf_transformer, 'test')
	target_train = data_train['especialist'].values
	target_test = data_test['especialist'].values

	clf = d_tree(attr_train,attr_test,target_train,target_test)
	return count_vect, tf_transformer, clf

def calculate_instance(instance, count_vect, tf_transformer, clf):
	instance = count_vect.transform([instance])
	instance = tf_transformer.transform(instance)
	return clf.predict(instance)

#Read data and split into train and test pandas dataframe
def read_dataset(file_name):
	data = pd.read_csv(file_name)
	data = data[['phrase','especialist']]
	return data

def d_tree(attr_train, attr_test, target_train, target_test):
	clf = tree.DecisionTreeClassifier()
	clf.fit(attr_train,target_train)	
	predicted = pd.DataFrame(clf.predict(attr_test))
	print(accuracy_score(target_test, predicted))	
	return clf

def r_forest(attr_train, attr_test, target_train, target_test):
	clf = RandomForestClassifier()
	clf.fit(attr_train,target_train)	
	predicted = pd.DataFrame(clf.predict(attr_test))
	print(f1_score(target_test, predicted,average='micro'))	

def mlp(attr_train, attr_test, target_train, target_test):
	clf=MLPClassifier()
	clf.fit(attr_train,target_train)	
	predicted = pd.DataFrame(clf.predict(attr_test))
	print(f1_score(target_test, predicted,average='micro'))	

def svm(attr_train, attr_test, target_train, target_test):
	clf=svm.SVC(probability=True)
	clf.fit(attr_train,target_train)	
	predicted = pd.DataFrame(clf.predict(attr_test))
	print(f1_score(target_test, predicted,average='micro'))	

def knn(attr_train, attr_test, target_train, target_test):
	clf=KNeighborsClassifier(n_neighbors=5)
	clf.fit(attr_train,target_train)	
	predicted = pd.DataFrame(clf.predict(attr_test))
	print(f1_score(target_test, predicted,average='micro'))	
	
def nb(attr_train, attr_test, target_train, target_test):
	clf=GaussianNB()
	clf.fit(attr_train,target_train)	
	predicted = pd.DataFrame(clf.predict(attr_test))
	print(f1_score(target_test, predicted,average='micro'))	

