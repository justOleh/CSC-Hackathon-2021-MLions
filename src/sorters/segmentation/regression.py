import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv
import time
import pickle
from sklearn.cluster import KMeans

def read_data(dataset_path:str, num_classes:int):
		X = []
		Y = []
		with open(dataset_path, newline='\n') as csv_file:
			reader = csv.reader(csv_file, delimiter=',')
			for row in reader:
				if len(row) == 0:
					continue
				row = np.array(row, dtype="float32")
				# first element is a class id and all the rest is a feature vector
				if int(row[0]) < num_classes:
					Y.append(int(row[0]))
					X.append(row[1:])
		return np.array(X), np.array(Y)


class LogRegression:
	CLASSES_DICT = ["People", "Contryside", "Urban", "Flowers", "Animals", "Art", "Docs", "Food"]

	def __init__(self):
		self.log_reg = None

	def fit(self, dataset_path:str):
		X, y = read_data(dataset_path, len(self.CLASSES_DICT))
		self.X_train, self.X_test, self.y_train, self.y_test = \
			train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
		self.classes_num = len(np.unique(self.y_test))
		# Initializing Logistic Regression object
		self.log_reg = LogisticRegression(n_jobs=-1)
		print("Training the model")
		t0 = time.time()
		self.log_reg.fit(self.X_train, self.y_train)
		print(f"Time, used fro training: {time.time() - t0} secs")
		# Calculating accuracy score on test dataset
		self.score = self.log_reg.score(self.X_test, self.y_test)
		print(f"Accuracy on test dataset: {self.score*100} %")

	def test_prediction(self):
		if self.log_reg is None:
			return
		indexes = np.random.randint(0, self.X_test.shape[0], size=10)
		print("Random test predictions:")
		for index in indexes:
			pred = self.log_reg.predict(self.X_test[index].reshape(1,-1))
			print('Predicted:', self.CLASSES_DICT[pred[0]], \
				'Actual:', self.CLASSES_DICT[self.y_test[index]])
	
	def predict(self, val):
		if self.log_reg is None:
			return
		return self.log_reg.predict(val)
	
	def get_class(self, id) -> str:
		if id < self.classes_num:
			return self.CLASSES_DICT[id]

	def confusion_matrix(self):
		if self.log_reg is None:
			return
		predictions = self.log_reg.predict(self.X_test)

		# Calculate confusion matrix
		cm = confusion_matrix(self.y_test, predictions)
		plt.figure(figsize=(15,15))
		ticklabels = self.CLASSES_DICT[0:self.classes_num]
		sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True,\
						yticklabels=ticklabels,  xticklabels=ticklabels)
		plt.ylabel('Actual label')
		plt.xlabel('Predicted label')
		all_sample_title = 'Accuracy Score: {0}'.format(self.score)
		plt.title(all_sample_title, size = 15)
		plt.show()

	def save(self, filename):
		# save the model to the filename
		pickle.dump(self.log_reg, open(filename, 'wb'))

	def load(self, filename):
		# load the model from the filename
		self.log_reg = pickle.load(open(filename, 'rb'))
		self.classes_num = len(self.CLASSES_DICT)



if __name__ == "__main__":
	DATASET_PATH = "dataset\labeled_dataset.csv"
	model = LogRegression()
	model.fit(DATASET_PATH)
	model.test_prediction()
	model.confusion_matrix()
