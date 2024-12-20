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

Downgrade pip and setuptools so that you can install slimevolley env:
```bash
pip install setuptools==65.5.0 pip==21
pip install slimevolleygym
pip install imageio
```

1. Start with installing all the requriements

```bash
pip install -r requirements.txt
pip install neat-python
pip install opencv-python
pip install jax
pip install pygame
pip install graphviz matplotlib
```

```
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

2. Run the main_v3.py

```bash
python main_v3.py 
```
```
apt-get update
apt-get install xvfb
apt-get install -y python3-opengl libglu1-mesa-dev freeglut3-dev
```
```
pip install --upgrade pip==23.0
pip install "gym==0.19.0"
pip install pyglet==1.5.27
```

```
xvfb-run -s "-screen 0 1400x900x24" python main_jax.py
```



