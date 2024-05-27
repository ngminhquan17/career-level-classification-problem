import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import FunctionTransformer
import re
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile

def format_string(x):
    result = re.findall("\,\s[A-Z]{2}$", x)
    if len(result) > 0:
        return result[0][2:]
    else:
        return x

data = pd.read_excel("final_project.ods", engine="odf", dtype=str)
data["location"] = data["location"].apply(format_string)
data = data.dropna(subset=["description"])

target = "career_level"
print(data[target].value_counts())
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify= y)
ros = SMOTEN(random_state=42, sampling_strategy={"director_business_unit_leader": 500, "specialist": 500, "managing_director_small_medium_company": 500 }, k_neighbors=2)
X_res, y_res = ros.fit_resample(x_train, y_train)


#Xử lý description bị khuyết 1 dòng
def preprocess_description(x):
    x = x.astype(str)  # Chuyển đổi tất cả giá trị thành chuỗi
    x = x.apply(lambda text: text.lower())  # Chuyển thành chữ thường
    return x

description_transfomer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('preprocess', FunctionTransformer(preprocess_description)),
    ('scaler', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_df=0.95, min_df=0.01))
])

preprocessor = ColumnTransformer(transformers=[
    ("title_features", TfidfVectorizer(), "title"),
    ("location_feature", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description_feature",  TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_df=0.95, min_df=0.01), "description"),
    ("function_feature", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_feature", TfidfVectorizer(), "industry"),
])

cls = Pipeline(steps=[
    ('preprocessor', preprocessor), #Tiền xử lý
    ('feature_selection', SelectKBest(chi2, k=800)),
    ('feature_selection', SelectPercentile(chi2, percentile=10)),
    ("model", RandomForestClassifier())
])

cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))

'''                           accuracy                           0.73      1615
                             macro avg       0.49      0.29      0.28      1615
                          weighted avg       0.70      0.73      0.69      1615'''