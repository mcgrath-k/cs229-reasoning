# CS229 Reasoning Project

## Information
---
Authors:
Aarohi Gupta (aarohig@stanford.edu)
Kathryn Lesh (leshk@stanford.edu)
Kyle McGrath (mcgrathk@stanford.edu)

Exploration on length generalization with a recursive self-attention based architecture.

### Dataset
---
Two datasets are available:

**Training / Validation:**
https://huggingface.co/datasets/mkyle/addition_dataset_5_million

**Test**
https://huggingface.co/datasets/mkyle/TEST_addition_dataset_5_million_scored


Modify the HF_REPO_ID in the codebase to:
```
mkyle/addition_dataset_5_million
```
```
mkyle/TEST_addition_dataset_5_million_scored
```
Depending on your usecase.

### Running a sample experiment:
```bash
python pretrain_simple.py batch_size=256 curriculum_type=diff_carries
```


## Setup
---
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
python pretrain_simple.py batch_size=256 curriculum_type=<null, diff_max_len, diff_min_len, diff_carries> 
```



