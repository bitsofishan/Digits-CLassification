from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
digits=load_digits()
print("Image Data shape",digits.data.shape)
print("Label Data shape",digits.target.shape)
plt.figure(figsize=(20,4))
for index,(image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title("Training: %i\n" %label,fontsize=10)

X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.23,random_state=2)
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)

predictions=logmodel.predict(X_test)
score=logmodel.score(X_test,y_test)
print(score)

cm=metrics.confusion_matrix(y_test,predictions)
print(cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,fmt=".3f",linewidth=.5,square=True,cmap="Blues_r");
plt.ylabel("Actual Label")
plt.xlabel("predicted Label")
plt.title("accuracy",score)





























