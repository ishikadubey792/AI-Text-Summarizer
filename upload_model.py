from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="saved_summary_model",
    repo_id="Ishikabharadwaj/text-summarizer-model",
    repo_type="model"
)

print("Model uploaded successfully!")