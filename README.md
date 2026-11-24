# Deep Learning: Basics → Computer Vision → Generative Models

A hands-on PyTorch course. It follows the classical ML course in `../machine-learning-learning`.

Each module in `modules/` has three notebooks:
- `lesson.ipynb`: a worked lesson. Run it top to bottom and experiment with it.
- `exercises.ipynb`: `# TODO` stubs, each followed by a self-check cell. When its asserts pass, the exercise is done.
- `solutions.ipynb`: reference answers. Try the exercises before opening these.

The approach is **intuition → a little math → build it from scratch → the PyTorch built-in**.

## Setup (Mac)
```bash
cd deep-learning-learning
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
jupyter lab
```
Every notebook's first cell picks the best device automatically: CUDA, then MPS, then CPU.

## Running on Colab (only for modules marked "Colab recommended")
1. Open https://colab.research.google.com, then File → Upload notebook.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run the first cell. It detects Colab and installs anything missing. These notebooks are self-contained.

## Roadmap and progress
Tick each module off as you finish it.

**Phase 0: Bridge**
- [ ] 00 PyTorch tensors and devices

**Phase 1: Neural network foundations**
- [ ] 01 Perceptron and MLP in NumPy
- [ ] 02 Backpropagation
- [ ] 03 Autograd from scratch
- [ ] 04 The PyTorch training loop
- [ ] 05 Optimization
- [ ] 06 Regularization and debugging

**Phase 2: Computer vision**
- [ ] 07 Convolutions from scratch
- [ ] 08 CNNs in PyTorch
- [ ] 09 Modern CNN architectures (ResNet)
- [ ] 10 Transfer learning
- [ ] 11 Interpreting CNNs (Grad-CAM)
- [ ] 12 Object detection *(Colab recommended)*
- [ ] 13 Segmentation with U-Net
- [ ] 14 Attention and Vision Transformers

**Phase 3: Generative models**
- [ ] 15 Autoencoders
- [ ] 16 Variational autoencoders
- [ ] 17 GANs *(Colab recommended)*
- [ ] 18 Advanced GANs *(Colab recommended)*
- [ ] 19 Diffusion from scratch *(Colab recommended)*
- [ ] 20 Guided and latent diffusion *(Colab recommended)*

**Phase 4: Capstone**
- [ ] 21 Capstone project

## For maintainers: rebuilding notebooks
Notebooks are generated from the sources in `tools/src/`:
```bash
python tools/build.py 03            # rebuild module 03
python tools/build.py all --execute # rebuild everything and run lesson + solutions quickly (FAST_DEV_RUN=1)
```
