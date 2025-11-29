# CS229 Reasoning Project

### Setup
---

For now, we will be using runpod.io as it is quick to set up and pricing is transparent. I am primarily using A40 GPU instances, they are cheaper and quick.

Here is my setup after runpod instance has been started.

SSH into runpod instance. Remember that any files outside of `/workspace` will be deleted when the instance is paused!

Install GitHub CLI (https://github.com/cli/cli/blob/trunk/docs/install_linux.md#debian)

```bash
(type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y
```

Log in:
```bash
gh auth login
```

Clone this repo:
```bash
cd /workspace
gh repo clone mcgrath-k/cs229-reasoning
```

Your template will come pre-installed with PyTorch & CUDA. To finish installing dependencies:
```bash
pip install wheel
pip install ninja
pip install flash-attn
python -m pip install -r /workspace/cs229-reasoning/requirements.txt
```

I had to run this to fix an error with adam-atan2
```bash
 pip uninstall adam-atan2 && pip install --verbose --no-cache-dir adam-atan2
```

Then login with your weights and biases account:

```bash
wandb login
```

Now test if you are able to train models:

```bash
# Download and build Sudoku dataset
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

# Start training (single GPU, smaller batch size)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```



