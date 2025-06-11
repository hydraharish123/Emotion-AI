import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import pandas as pd

y_labelled = np.load('emotion_labels.npy')
X_labelled_pca = np.load('X_labelled_pca.npy')


class SVMClassifier:
    def __init__(self, kernel='poly', param_grid=None, cv=5, scoring='accuracy', probability=False):
        self.param_grid = param_grid or {
            'svc__degree': [2, 3, 4],
            'svc__coef0': [0, 1, 5],
            'svc__C': [0.1, 1, 10]
        }
        self.pipeline = Pipeline([
            ('svc', SVC(kernel=kernel, class_weight='balanced', probability=probability))
        ])

        self.grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=cv, scoring=scoring)
        self.best_model = None

    def fit(self, X, y):
        self.grid_search.fit(X, y)
        self.best_model = self.grid_search.best_estimator_
        print("Best Parameters:", self.grid_search.best_params_)
        print("Best Score:", self.grid_search.best_score_)

    def predict(self, X):
        if self.best_model is None:
            raise Exception("Model has not been trained. Call `.fit(X, y)` first.")
        return self.best_model.predict(X)
    
    def report(self, X, y_true):
        y_pred = self.predict(X)
        print("Classification Report:\n", classification_report(y_true, y_pred))


## Model 1 Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_labelled_pca, y_labelled, test_size=0.3, random_state=42,  stratify=y_labelled)

svm_clf = SVMClassifier(cv=10, param_grid = {
    'svc__C': [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90,  100],       
    'svc__gamma': ['scale', 0.001, 0.01, 0.1, 1] 
}, kernel='rbf', scoring='f1_macro')

svm_clf.fit(X=X_train, y=y_train)
y_pred = svm_clf.predict(X=X_test)
svm_clf.report(X=X_test, y_true=y_test)

print(f"Model 1 Complete")
print(confusion_matrix(y_pred=y_pred, y_true=y_test))

## Model 2 Training
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_labelled_pca, y_labelled, test_size=0.3, random_state=42,  stratify=y_labelled)

svm_clf_smote = SVMClassifier(cv=10, param_grid = {
    'svc__C': [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90,  100],       
    'svc__gamma': ['scale', 0.001, 0.01, 0.1, 1] 
}, kernel='rbf', scoring='f1_macro')

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_smote, y_train_smote) ## shape=(3320, 68)) 

svm_clf_smote.fit(X=X_train_resampled, y=y_train_resampled)
y_pred_smote = svm_clf_smote.predict(X=X_test_smote)

svm_clf_smote.report(X=X_test_smote, y_true=y_test_smote)

print(f"Model 2 Complete")
print(confusion_matrix(y_pred=y_pred_smote, y_true=y_test_smote))


## Model 3 training
## Using a subset 

df = pd.DataFrame(X_labelled_pca)
df['label'] = y_labelled

neutral_df = df[df['label'] == 'Neutral']
minority_df = df[df['label'] != 'Neutral']

# Downsample Neutral to match the minority size
neutral_downsampled = resample(neutral_df, 
                               replace=False, 
                               n_samples=len(minority_df), 
                               random_state=42)

balanced_subset = pd.concat([neutral_downsampled, minority_df])
balanced_subset = balanced_subset.sample(frac=1, random_state=42)  

X_subset = balanced_subset.drop('label', axis=1).values
y_subset = balanced_subset['label'].values

print(f"The dimensions of input subset is {X_subset.shape}")
print(f"The dimensions of output subset is {y_subset.shape}")

X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(X_subset, y_subset, test_size=0.3, random_state=42,  stratify=y_subset)

svm_clf_subset = SVMClassifier(cv=10, param_grid = {
    'svc__C': [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90,  100],       
    'svc__gamma': ['scale', 0.001, 0.01, 0.1, 1] 
}, kernel='rbf', scoring='f1_macro')

svm_clf_subset.fit(X=X_train_subset, y=y_train_subset)

svm_clf_subset.report(X=X_test_subset, y_true=y_test_subset)

y_pred_subset = svm_clf_subset.predict(X=X_test_subset)

print(f"Model 3 Complete")
print(confusion_matrix(y_pred=y_pred_subset, y_true=y_test_subset))


## Model 4 Training
print("Model 4 => Subset + SMOTE")

X_train_sub_smote, X_test_sub_smote, y_train_sub_smote, y_test_sub_smote = train_test_split(X_subset, y_subset, test_size=0.3, random_state=42,  stratify=y_subset)

## _sub_smote
smote = SMOTE(random_state=42)
X_train_subset_Smote, y_train_subset_Smote = smote.fit_resample(X_train_sub_smote, y_train_sub_smote) ##  shape=(1832, 68)) and shape=(1832,)

svm_clf_sub_smote = SVMClassifier(cv=10, param_grid = {
    'svc__C': [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90,  100],       
    'svc__gamma': ['scale', 0.001, 0.01, 0.1, 1] 
}, kernel='rbf', scoring='f1_macro')

y_pred_subset_smote = svm_clf_sub_smote.predict(X=X_test_sub_smote)

svm_clf_sub_smote.report(X=X_test_sub_smote, y_true=y_test_sub_smote)

print(f"Model 4 Complete")
print(confusion_matrix(y_pred=y_pred_subset_smote, y_true=y_test_sub_smote))

## Model 5
print("Model 5 => Using 2 models to give a prediction")

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_labelled_pca, y_labelled, stratify=y_labelled, test_size=0.3)

svm1_x = X_train_final
svm1_y = np.array(['NotNeutral' if val != 'Neutral' else 'Neutral' for val in y_train_final])

X_train, X_test, y_train, y_test = train_test_split(svm1_x, svm1_y, test_size=0.3, random_state=42,  stratify=svm1_y)

svm_1 = SVMClassifier(cv=10, param_grid = {
    'svc__C': [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90,  100],       
    'svc__gamma': ['scale', 0.001, 0.01, 0.1, 1] 
}, kernel='rbf', scoring='f1_macro', probability=True)

svm_1.fit(X=X_train, y=y_train)

svm_1.report(X=X_test, y_true=y_test)

df1 = pd.DataFrame(X_train_final)
df1['label'] = y_train_final ## 644 rows × 69 columns

other_df = df1[df1['label'] != 'Neutral'] ## 229 rows × 69 columns

other_df = other_df.sample(frac=1, random_state=42) 

X_subset = other_df.drop('label', axis=1).values
y_subset = other_df['label'].values

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_subset, y_subset, test_size=0.3, random_state=42,  stratify=y_subset)

svm_2 = SVMClassifier(cv=10, param_grid = {
    'svc__C': [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90,  100],       
    'svc__gamma': ['scale', 0.001, 0.01, 0.1, 1] 
}, kernel='rbf', scoring='f1_macro')

svm_2.fit(X=X_train_2, y=y_train_2)

svm_2.report(X=X_test_2, y_true=y_test_2)

final_pred = []

for i in range(len(X_test_final)):
    x = X_test_final[i].reshape(1, -1)

    pred1 = svm_1.predict(x)[0]

    if(pred1 == "Neutral"):
        final_pred.append("Neutral")
    else:
        pred2 = svm_2.predict(x)[0]
        final_pred.append(pred2)

print(classification_report(y_pred=final_pred, y_true=y_test_final))

print(f"Model 5 Complete")
print(confusion_matrix(y_pred=final_pred, y_true=y_test_final))