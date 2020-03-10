import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as py_plot


data = pd.read_csv("car.data")
# print(data.head())
# buying,maint,door,persons,lug_boot,safety,class
le = preprocessing.LabelEncoder()

buying = le.fit_transform(data["buying"])
maint = le.fit_transform(data["maint"])
door = le.fit_transform(data["door"])
persons = le.fit_transform(data["persons"])
lug_boot = le.fit_transform(data["lug_boot"])
safety = le.fit_transform(data["safety"])
cls = le.fit_transform(data["class"])

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# print(x_train, "\n", x_test)


model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

names = ["unacc", 'acc', 'good', 'very_good']

prediction = model.predict(x_test)

for i in range(len(prediction)):
    print("Prediction : ", names[prediction[i]], " Test data : ", x_test[i], "Actual value : ", names[y_test[i]])

    n = model.kneighbors([x_test[i]], 9, True)
    print(n)



