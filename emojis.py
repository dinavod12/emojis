import pandas as pd
df = pd.read_csv("Desktop/train.txt/train.txt",sep=';',names=['Words', 'Emotions'])
import re 
import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(16000):
    review = re.sub("[^a-zA-Z]"," ",df["Words"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    review = [ps.stem(word) for word in review  if not word in set(all_stopwords)]
    review = " ".join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
model2 = RandomForestClassifier(n_estimators = 20, criterion = "entropy", random_state = 2)
model2.fit(X_train,Y_train)
y_pred = model2.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,y_pred))
emotions = input("input your emotions")
inp = cv.transform([a]).toarray()
emoji_dist={"anger":"Desktop/emojis/angry.png","fear":"Desktop/emojis/fearful.png","love":"Desktop/emojis/love.jpg","joy":"Desktop/emojis/happy.png","sadness":"Desktop/emojis/sad.png","surprise":"Desktop/emojis/surpriced.png"}
pridict = model2.predict(inp)
import cv2 as cv1
img = cv1.imread(emoji_dist[pridict[0]])
cv1.imshow(f"{pridict[0]}",img)
cv1.waitKey(0);                         
cv1.destroyAllWindows();
