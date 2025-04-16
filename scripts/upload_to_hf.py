from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_to_huggingface(file_path, repo_id, token):
    """
    Upload a .pt file to Hugging Face Hub
    
    Args:
        file_path (str): Path to the .pt file
        repo_id (str): Hugging Face repository ID (format: 'username/repo-name')
        token (str): Hugging Face API token
    """
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id, token=token, exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Upload the file
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            token=token
        )
        print(f"Successfully uploaded {file_path} to {repo_id}")
    except Exception as e:
        print(f"Error uploading file: {e}")

if __name__ == "__main__":
    # Get Hugging Face token from environment variable
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("Please set your HUGGINGFACE_TOKEN in the .env file")
        exit(1)
    
    # Example usage
    repo_id = "your-username/your-repo-name"  # Replace with your repository name
    pt_files = [
        "../data/training_data.pt",
        # Add other .pt files here
    ]
    
    for file_path in pt_files:
        if os.path.exists(file_path):
            upload_to_huggingface(file_path, repo_id, token)
        else:
            print(f"File not found: {file_path}") 