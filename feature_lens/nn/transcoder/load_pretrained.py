import pathlib
from huggingface_hub import snapshot_download
from .transcoder import Transcoder
import sys

# Add parent directory to sys.path
# This is to ensure 'sae_training' is a discoverable module
# which is required for loading the transcoder checkpoints using torch.load
project_dir = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir))

REPO_ID = "pchlenski/gpt2-transcoders"

FILENAMES = [
    f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.ln2.hook_normalized_24576.pt"
    for layer in range(12)
]


def download(filenames: list[str] = FILENAMES) -> list[pathlib.Path]:
    folder_path = pathlib.Path(
        snapshot_download(repo_id=REPO_ID, allow_patterns="*.pt")
    )
    file_paths = [folder_path / filename for filename in filenames]
    return file_paths


def load_pretrained(filenames: list[str] = FILENAMES) -> dict[str, Transcoder]:
    file_paths = download(filenames)
    transcoders = {}
    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        transcoder = Transcoder.load_from_pretrained(str(file_path))
        transcoders[file_path.stem] = transcoder
    return transcoders


if __name__ == "__main__":
    transcoders = load_pretrained()
    print(transcoders)
