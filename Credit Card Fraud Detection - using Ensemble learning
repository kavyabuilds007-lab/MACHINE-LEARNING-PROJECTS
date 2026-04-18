#IMPORTS
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier

from imblearn.over_sampling import SMOTE


#LOAD DATASET
file_path = '/content/drive/MyDrive/my_project/creditcard.csv'
df = pd.read_csv(file_path)
df = df.groupby('Class', group_keys=False).apply(
    lambda x: x.sample(frac=0.2, random_state=42)
)
print("Dataset Shape:", df.shape)
print("\nClass Distribution:\n", df['Class'].value_counts())
print("\nFraud Percentage:", (df['Class'].sum() / len(df)) * 100)
print("\nSample Data:")
df.head()


#FEATURE-TARGET SPLIT AND SCALING
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])


#MUTUAL INFORMATION - FEATURE SELECTION
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI Score': mi_scores
}).sort_values(by='MI Score', ascending=False)

print("\nTop Features based on Mutual Information:\n")
print(mi_df.head(10))
top_features = mi_df['Feature'].head(20).values
print("\nSelected Features:\n", top_features)
X = X[top_features]


#TRAIN-TEST SPLIT OF DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


#HANDLING CLASS IMBALANCE USING SMOTE
smote = SMOTE(random_state=42)
print("\nBefore SMOTE:\n", y_train.value_counts())
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE:\n", y_train_sm.value_counts())


#BASE MODELS
lr = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier()
svm = SVC(kernel='linear',probability=True)
dt = DecisionTreeClassifier(max_depth=10)

#STACK MODEL
estimators = [
    ('knn', knn),
    ('svm', svm),
    ('dt', dt)
]
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    passthrough=False
)


#TRAINING THE MODELS
print("\nTraining Base Models...")
lr.fit(X_train_sm, y_train_sm)
knn.fit(X_train_sm, y_train_sm)
dt.fit(X_train_sm, y_train_sm)
svm.fit(X_train_sm, y_train_sm)

print("\nTraining Stacking Model...")
stack_model.fit(X_train_sm, y_train_sm)
print("Stacking Model Trained")


#EVALUATING THE MODELS
def evaluate(model, name, threshold=0.2):
    print(f"\n===== {name} =====")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

evaluate(lr, "Logistic Regression")
evaluate(knn, "KNN")
evaluate(svm, "SVM")
evaluate(dt, "Decision Tree")
evaluate(stack_model, "Stacking Model")


#RESULTS AND PLOTS 
-->1. FEATURE IMPORTANCE
mi_df_sorted = mi_df.sort_values(by='MI Score', ascending=False).head(15)

plt.figure()
plt.barh(mi_df_sorted['Feature'], mi_df_sorted['MI Score'])
plt.gca().invert_yaxis()
plt.xlabel("MI Score")
plt.ylabel("Features")
plt.title("Feature Importance (Mutual Information)")
plt.show()

-->2. PRECISION VS RECALL PLOT
y_prob = stack_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall vs Threshold")
plt.legend()
plt.show()

-->3. ROC AND P-R CURVES
fpr, tpr, _ = roc_curve(y_test, y_prob)
precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
pr_auc = auc(recall_pr, precision_pr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot(recall_pr, precision_pr, label=f"PR Curve (AUC = {pr_auc:.3f})")
plt.xlabel("X-axis (FPR / Recall)")
plt.ylabel("Y-axis (TPR / Precision)")
plt.title("ROC vs Precision-Recall Curve")
plt.legend()
plt.show()

-->4. CONFUSION MATRIX
threshold = 0.2
y_pred = (y_prob >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha='center', va='center')
plt.show()

-->5. TEST INSTANCE
print("\n===== SINGLE TRANSACTION TEST =====")
sample = X_test.iloc[0:1]
probability = stack_model.predict_proba(sample)[0][1]

threshold = 0.2
prediction = 1 if probability >= threshold else 0

print("Fraud Probability:", probability)
print("Prediction:", "FRAUD" if prediction == 1 else "NOT FRAUD")
