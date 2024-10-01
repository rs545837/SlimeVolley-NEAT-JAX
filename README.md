# SlimeVolley-NEAT-JAX

## Quick Start

Start with setting up miniconda3 in your remote linux gpu server. Copy and paste the lines below to do so. 
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

Create and activate new env:
```bash
conda create --name rl_test -y
conda activate rl_test
```

1. Start with installing all the requriements

```bash
pip install -r requirements.txt
```

2. Run the main.py

```bash
python main.py 
```



