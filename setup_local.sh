(type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y


# gh repo clone mcgrath-k/cs229-reasoning

pip install wheel
pip install ninja
# pip install flash-attn
#  uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.18/flash_attn-2.8.3+cu130torch2.9-cp310-cp310-linux_x86_64.whl
python -m pip install -r requirements.txt
pip uninstall -y adam-atan2
pip install --verbose --no-cache-dir adam-atan2


