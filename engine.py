import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from lime.lime_tabular import LimeTabularExplainer

DATA_PATH = "data/adult.csv"


def load_data(sample_size=3000):
    """Loads and samples the dataset."""
    df = pd.read_csv(DATA_PATH, header=None)
    cols = [
        'age','workclass','fnlwgt','education','education-num','marital-status',
        'occupation','relationship','race','sex','capital-gain','capital-loss',
        'hours-per-week','native-country','income'
    ]
    df.columns = cols
    df = df.dropna().sample(sample_size, random_state=42).reset_index(drop=True)
    return df

def preprocess(df, sensitive_col="sex"):
    """Encodes features and returns X, y, and the sensitive attribute."""
    X = pd.get_dummies(df.drop(columns=["income"]), drop_first=True)
    y = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)
    sensitive = df[sensitive_col]
    return X, y, sensitive

def train_and_evaluate(X, y, sensitive):
    """Trains logistic model, computes fairness metrics and trust score."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=400).fit(X_train, y_train)
    preds = model.predict(X_test)

    sp = demographic_parity_difference(y_test, preds, sensitive_features=sensitive.loc[y_test.index])
    eo = equalized_odds_difference(y_test, preds, sensitive_features=sensitive.loc[y_test.index])

    trust = max(0, min(100, 100 - (abs(sp) + abs(eo)) * 50))
    return model, X_train, X_test, y_test, sp, eo, trust

def explain(model, X_train, instance):
    """Generates a local explanation for one prediction using LIME."""
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=["<=50K", ">50K"],
        mode="classification"
    )
    exp = explainer.explain_instance(instance.values[0], model.predict_proba)
    return exp.as_list()

