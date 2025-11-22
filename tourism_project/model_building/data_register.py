from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "karanjkarnishi/Tourism-Package-Prediction"
repo_type = "dataset"

# Initialize API client
token = os.getenv("HF_TOKEN")

if token is None or token == '':
    print("HF_TOKEN is not set!")
else:
    print(f"HF_TOKEN is set. Using token for authentication.")
    
api = HfApi(token)

print(f"HF_TOKEN :  '{token}' exists. Using it.")

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
