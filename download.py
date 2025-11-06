from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="cartesia-ai/hnet_2stage_L", local_dir="/data/hf_cache/hnet_2stage_L"
)
