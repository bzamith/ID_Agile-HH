import argparse #Import Argsparse
import pandas as pd #For datasets
import statistics #For mean and standard deviation
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

#Read data and split into train and test pandas dataframe
def read_dataset(file_name):
	data = pd.read_csv(file_name)
	print(data)
	for i in range(0,len(data.columns)):
		data.iloc[i].replace('No','False')
	training_data, test_data = train_test_split(data, train_size=0.85, shuffle=True)
	return training_data,test_data

def ml_inputs(training_data,test_data):
	attr=training_data.copy()
	target=training_data.copy()
	del attr['readmitted']
	target=target['readmitted'] 
	real = test_data.copy()
	predicted = test_data.copy()
	del test_data['readmitted']
	real = real['readmitted']
	return attr,target,real

def d_tree(attr, target, real, test_data):
	clf = tree.DecisionTreeClassifier()
	clf.fit(attr,target)	
	#predicted = pd.DataFrame(clf.predict_proba(test_data))
	predicted = pd.DataFrame(clf.predict(test_data))
	#print(accuracy_score(real, predicted))
	print(f1_score(real, predicted))	

def r_forest(attr,target,real, test_data):
	clf = RandomForestClassifier()
	clf.fit(attr,target)
	predicted=pd.DataFrame(clf.predict(test_data))
	#print(accuracy_score(real, predicted))
	print(f1_score(real, predicted))	

def mlp(attr,target,real, test_data):
	clf=MLPClassifier()
	clf.fit(attr,target)
	predicted=pd.DataFrame(clf.predict(test_data))
	#print(accuracy_score(real, predicted))
	print(f1_score(real, predicted))

def svm(attr,target,real, test_data):
	clf=svm.SVC(probability=True)
	clf.fit(attr,target)
	predicted=pd.DataFrame(clf.predict(test_data))
	#print(accuracy_score(real, predicted))
	print(f1_score(real, predicted))

def knn(attr,target,real, test_data):
	clf=KNeighborsClassifier(n_neighbors=5)
	clf.fit(attr,target)
	predicted=pd.DataFrame(clf.predict(test_data))
	#print(accuracy_score(real, predicted))
	print(f1_score(real, predicted))
	
def nb(attr,target,real, test_data):
	clf=GaussianNB()
	clf.fit(attr,target)
	predicted=pd.DataFrame(clf.predict(test_data))
	#print(accuracy_score(real, predicted))
	print(f1_score(real, predicted))

#Main Function
if __name__ == '__main__':
	#Args parse to get file name
	parser = argparse.ArgumentParser(description='Please provide the dataset file name', add_help=True)
	parser.add_argument('-i','--input', dest='inputFile', metavar='inputFile', type=str, help='Dataset', required=True)
	args = parser.parse_args()
	file_name = args.inputFile

	#Execution
	training_data,test_data = read_dataset(file_name)
	attr, target, real = ml_inputs(training_data,test_data)



	print("**DTREE")
	d_tree(attr,target,real,test_data)
	print("**RFOREST")
	r_forest(attr,target,real,test_data)
	print("**MLP")
	mlp(attr,target,real,test_data)
	print("**KNN")
	knn(attr,target,real,test_data)
	print("**NB")
	nb(attr,target,real,test_data)
