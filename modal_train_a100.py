import modal
import modal.experimental
import os
from pathlib import Path

# Define the Modal app
app = modal.App("nano-gpt-1")
path_repo = "/root/modded-nanogpt"

commands = f'''
git clone -b poc_4090_dev https://github.com/broqdev/modded-nanogpt.git {path_repo}
cd {path_repo} && pip install -r requirements.txt
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
# downloads only the first 800M training tokens to save time
cd {path_repo} && python data/cached_fineweb10B.py 8
# ./run.sh
'''
commands = [cmd for cmd in commands.splitlines() if cmd.strip() and not cmd.strip().startswith("#")]

# Create the image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(*commands)
)

# Define the main function
@app.function(
    image=image,
    gpu="A100:1",
    timeout=20 * 60,  # set 20 minutes timeout since modal may take long time to request gpus
)
@modal.experimental.clustered(size=1)
def train_nanogpt():
    assert Path(f"{path_repo}/train_gpt_a100.py").exists()

    # change working directory to repo
    os.chdir(path_repo)

    # import the 'torchrun' interface directly.
    from torch.distributed.run import parse_args, run

    cluster_info = modal.experimental.get_cluster_info()

    run(
        parse_args(
            [
                f"--standalone",
                f"--nproc-per-node=1",
                "train_gpt_a100.py",
            ]
        )
    )