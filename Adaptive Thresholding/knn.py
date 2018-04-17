from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


X = labeled_df.drop(['picked'], axis=1).values
y = labeled_df[['picked']].values

y = np.ravel(y) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


knn = KNeighborsClassifier(n_neighbors = 17).fit(X_train, y_train)
 
# accuracy on X_test
accuracy = knn.score(X_test, y_test)
print(accuracy)