from huggingface_hub import hf_hub_download, list_repo_files
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_from_huggingface(filename, repo_id, token):

    hf_hub_download(repo_id=repo_id, 
                    filename=filename, 
                    local_dir="tensors/", 
                    token=token, 
                    repo_type="dataset")

if __name__ == "__main__":
    token = os.getenv("HUGGINGFACE_TOKEN")
    repo_id = "titaneve/hackernews_prediction_mlx"

    download_from_huggingface("eve_corpus.pt", repo_id, token)
    download_from_huggingface("eve_id_to_word.pt", repo_id, token)
    download_from_huggingface("eve_training_data.pt", repo_id, token)
    download_from_huggingface("eve_word_to_id.pt", repo_id, token)

    print("Download eve_corpus.pt successful")
    print("Download eve_id_to_word.pt successful")
    print("Download eve_training_data.pt successful")
    print("Download eve_word_to_id.pt successful")  
