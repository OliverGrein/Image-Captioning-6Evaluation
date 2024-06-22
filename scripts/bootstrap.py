import os
import shutil
from huggingface_hub import HfApi, hf_hub_download


def download_folder(repo_id, folder_path, save_dir):
    os.makedirs(os.path.join(save_dir, folder_path), exist_ok=True)
    api = HfApi()

    try:
        folder_contents = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        for file_path in folder_contents:
            if file_path.startswith(folder_path):
                try:
                    file_name = os.path.basename(file_path)
                    downloaded_file_path = hf_hub_download(
                        repo_id=repo_id, filename=file_path, repo_type="dataset"
                    )
                    save_path = os.path.join(save_dir, folder_path, file_name)
                    shutil.copy2(downloaded_file_path, save_path)
                    print(f"Successfully downloaded {file_name} to {save_path}")
                except Exception as e:
                    print(f"Error downloading {file_name}: {str(e)}")
    except Exception as e:
        print(f"Error accessing folder {folder_path}: {str(e)}")


def main():
    repo_id = "Spycner/caption_embeddings"
    save_dir = "data"

    # List of folders to download
    folders = ["cohere", "openai", "vertexai", "voyageai"]

    for folder in folders:
        download_folder(repo_id, folder, save_dir)


if __name__ == "__main__":
    main()
