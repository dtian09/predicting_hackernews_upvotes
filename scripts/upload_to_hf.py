from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
import glob

# Load environment variables from .env file
load_dotenv()

def main():
    # Get the token from environment variables
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)

    #pt_files = glob.glob("./data/*.pt")
    pt_files = ("./tensors/cbow_model_dim100_batch32_final.pt", "./tensors/word_embeddings_dim100.pt")

    for file in pt_files:
        print(file)
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=os.path.basename(file),
            repo_id="titaneve/hackernews_prediction_mlx",
            repo_type="dataset",
            token=token
        )
        print(f"Successfully uploaded {file}")


if __name__ == "__main__":
    main() 