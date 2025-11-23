# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Set MLflow tracking URI to a temporary directory for local tracking within Colab
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism_project-training-experiment")

Xtrain_path = "hf://datasets/karanjkarnishi/Tourism-Package-Prediction-Dataset/Xtrain.csv"
Xtest_path = "hf://datasets/karanjkarnishi/Tourism-Package-Prediction-Dataset/Xtest.csv"
ytrain_path = "hf://datasets/karanjkarnishi/Tourism-Package-Prediction-Dataset/ytrain.csv"
ytest_path = "hf://datasets/karanjkarnishi/Tourism-Package-Prediction-Dataset/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# List of numerical features in the dataset
numeric_features = [
    'Age',                      # Age of the customer
    'CityTier',                 # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
    'DurationOfPitch',          # Duration of the sales pitch delivered to the customer
    'NumberOfPersonVisiting',   # Total number of people accompanying the customer on the trip
    'NumberOfFollowups',        # Total number of follow-ups by the salesperson after the sales pitch
    'PreferredPropertyStar',    # Preferred hotel rating by the customer
    'NumberOfTrips',            # Average number of trips the customer takes annually
    'Passport',                 # Whether the customer holds a valid passport (0: No, 1: Yes)
    'PitchSatisfactionScore',   # Score indicating the customer's satisfaction with the sales pitch
    'OwnCar',                   # Whether the customer owns a car (0: No, 1: Yes)
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer
    'MonthlyIncome'             # Gross monthly income of the customer
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',            # The method by which the customer was contacted (Company Invited or Self Inquiry)
    'Occupation',               # Customer's occupation (e.g., Salaried, Freelancer)
    'Gender',                   # Gender of the customer (Male, Female)
    'MaritalStatus',            # Marital status of the customer (Single, Married, Divorced)
    'Designation',              # Customer's designation in their current organization
    'ProductPitched'            # The type of product pitched to the customer
]


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print ('Class Weight : ', class_weight)

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42
)

# Pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgb", xgb_model)
])

# Define hyperparameter grid
# Hyperparameter Space
param_dist = {
    "xgb__n_estimators": [100, 150, 200, 250],
    "xgb__max_depth": [3, 4, 5],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__subsample": [0.7, 0.8, 0.9],
    "xgb__colsample_bytree": [0.5, 0.7, 1.0],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow
with mlflow.start_run():

    mlflow.log_param("class_weight", class_weight)

    # Randomized Search (faster than Grid)
    search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=2
    )

    search.fit(Xtrain, ytrain)

    # Best model
    best_model = search.best_estimator_
    mlflow.log_params(search.best_params_)

    # Predictions
    threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test  = (best_model.predict_proba(Xtest)[:, 1]  >= threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest,  y_pred_test,  output_dict=True)

    # Log Metrics
    for prefix, rep in [("train", train_report), ("test", test_report)]:
        mlflow.log_metric(f"{prefix}_accuracy", rep["accuracy"])
        mlflow.log_metric(f"{prefix}_precision", rep["1"]["precision"])
        mlflow.log_metric(f"{prefix}_recall", rep["1"]["recall"])
        mlflow.log_metric(f"{prefix}_f1", rep["1"]["f1-score"])

    # Save model locally
    model_path = "tourism_prediction_model_v2.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "karanjkarnishi/tourism_package_prediction_model"
    repo_type = "model"

    token=os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    # Step 1: Check if the space exists, otherwise create it
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # Step 2: Upload the model artifact to the space
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type="model"
    )
    
    print("Model uploaded to HF successfully.")
