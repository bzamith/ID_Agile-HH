import argparse #Import Argsparse
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

def pln(data):
	count_vect = CountVectorizer()
	tf_transformer = TfidfTransformer()

	data_vect = count_vect.fit_transform(data["phrase"].values)
	data_vect = tf_transformer.fit_transform(data_vect)

	attr_train, attr_test = train_test_split(data_vect,test_size=0.2,shuffle=False)
	target_train, target_test = train_test_split(data["especialist"],test_size=0.2,shuffle=False)
	target_test = target_test.values

	r_forest(attr_train,attr_test,target_train,target_test)

#nltk stop words

#Read data and split into train and test pandas dataframe
def read_dataset(file_name):
	data = pd.read_csv(file_name)
	data = data[['phrase','especialist']]
	return data

def d_tree(attr_train, attr_test, target_train, target_test):
	clf = tree.DecisionTreeClassifier()
	clf.fit(attr_train,target_train)	
	predicted = pd.DataFrame(clf.predict(attr_test))
	print(f1_score(target_test, predicted,average='micro'))	

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

#Main Function
if __name__ == '__main__':
	#Args parse to get file name
	parser = argparse.ArgumentParser(description='Please provide the dataset file name', add_help=True)
	parser.add_argument('-i','--input', dest='inputFile', metavar='inputFile', type=str, help='Dataset', required=True)
	args = parser.parse_args()
	file_name = args.inputFile

	#Execution
	data = read_dataset(file_name)
	pln(data)
