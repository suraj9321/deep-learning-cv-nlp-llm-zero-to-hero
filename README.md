# Deep Learning: Basics → Computer Vision → Generative Models → NLP & LLMs → Multimodal AI

A hands-on PyTorch course. It follows the classical ML course in `../machine-learning-learning`.

Each module in `modules/` has three notebooks:

- `lesson.ipynb`: a worked lesson. Run it top to bottom and experiment with it.
- `exercises.ipynb`: `# TODO` stubs, each followed by a self-check cell. When its asserts pass, the exercise is done.
- `solutions.ipynb`: reference answers. Try the exercises before opening these.

The approach is **intuition → a little math → build it from scratch → the PyTorch built-in**.

## Setup (Mac)

```bash
cd deep-learning-cv-nlp-llm-zero-to-hero
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
jupyter lab
```

Every notebook's first cell picks the best device automatically: CUDA, then MPS, then CPU.

Optional - Classic ML before Deep Learning 
**Classical ML**: [`classical-machine-learning-basic-to-advanced-in-steps`](https://github.com/suraj9321/classical-machine-learning-basic-to-advanced-in-steps)

## Running on Colab (only for modules marked "Colab recommended")

1. Open https://colab.research.google.com, then File → Upload notebook.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run the first cell. It detects Colab and installs anything missing. These notebooks are self-contained.

## Roadmap and progress

Tick each module off as you finish it. Status: ✅ built · 🚧 not built yet. _(Colab recommended)_ marks the
modules that are slow on a laptop. Under each module, 🎥 lists recommended videos and 📄 recommended papers
(⭐ = start here; the others are optional). The same lists appear at the end of each lesson notebook, with a note on
why each one is worth your time. Click a phase's summary line to expand its modules.

**Phases at a glance**

1. [Phase 0: Bridge from classical ML](#phase-0-bridge-from-classical-ml-) ✅ (module 00)
2. [Phase 1: Neural network foundations](#phase-1-neural-network-foundations-) ✅ (modules 01–06)
3. [Phase 2: Computer vision](#phase-2-computer-vision-) 🚧 (modules 07–14)
4. [Phase 3: Generative models](#phase-3-generative-models-) 🚧 (modules 15–20)
5. [Phase 4: NLP and large language models](#phase-4-nlp-and-large-language-models-) 🚧 (modules 21–28)
6. [Phase 5: Multimodal AI](#phase-5-multimodal-ai-) 🚧 (modules 29–32)
7. [Phase 6: Capstone](#phase-6-capstone-) 🚧 (module 33)

### Phase 0: Bridge from classical ML ✅

<details>
<summary>Module 00: PyTorch tensors, shapes and devices</summary>

- [ ] **00 PyTorch tensors and devices**
  - Creating tensors; converting between NumPy and PyTorch (when memory is shared)
  - dtypes: why float32 for data and long for labels
  - Indexing, broadcasting and reductions (`dim`, `keepdim`)
  - Shapes: the (N, C, H, W) layout, `view`/`reshape`/`permute`/`unsqueeze`/`flatten`, contiguity
  - Devices: CPU, CUDA and MPS, `.to(device)`, device-mismatch errors, CPU vs GPU timing
  - Batched matrix multiply and `einsum`; in-place operations
  - 📄 **Recommended reading**
    - [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703) (Paszke et al., 2019)

</details>

### Phase 1: Neural network foundations ✅

<details>
<summary>Modules 01–06: perceptron, backprop, autograd, training loop, optimization, regularization</summary>

- [ ] **01 Perceptron and MLP in NumPy**
  - A neuron as weighted sum + activation; sigmoid, tanh, ReLU
  - The perceptron learning rule, and why it can't learn XOR
  - Hidden layers as learned representations; why non-linearity is needed
  - A batched MLP forward pass, He initialisation, binary cross-entropy
  - Training with numerical gradients, and why that doesn't scale
  - 🎥 **Recommended videos**
    - ⭐ [Neural networks, Chapters 1–2: 'But what is a neural network?' and 'Gradient descent'](https://www.3blue1brown.com/lessons/gradient-descent) (3Blue1Brown)
    - [Introduction to Deep Learning (lecture 1)](https://introtodeeplearning.com/) (MIT 6.S191)
  - 📄 **Recommended reading**
    - [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://doi.org/10.1037/h0042519) (Rosenblatt, 1958)
    - [Multilayer Feedforward Networks are Universal Approximators](https://doi.org/10.1016/0893-6080(89)90020-8) (Hornik, Stinchcombe & White, 1989)
    - *Rectified Linear Units Improve Restricted Boltzmann Machines* (Nair & Hinton, 2010)
- [ ] **02 Backpropagation**
  - Computational graphs and the chain rule ("upstream × local")
  - Gradients for linear layers, ReLU, sigmoid + BCE, softmax + cross-entropy
  - Backprop through a full MLP with cached forward values
  - Gradient checking against numerical gradients
  - Multi-class classification on a spiral; a first look at vanishing gradients
  - 🎥 **Recommended videos**
    - ⭐ [What is backpropagation really doing? (Chapter 3)](https://www.3blue1brown.com/lessons/backpropagation) (3Blue1Brown)
    - ⭐ [Backpropagation calculus (Chapter 4)](https://www.3blue1brown.com/lessons/backpropagation-calculus) (3Blue1Brown)
    - [makemore Part 4: Becoming a Backprop Ninja](https://karpathy.ai/zero-to-hero.html) (Andrej Karpathy)
  - 📄 **Recommended reading**
    - ⭐ [Learning Representations by Back-propagating Errors](https://doi.org/10.1038/323533a0) (Rumelhart, Hinton & Williams, 1986)
- [ ] **03 Autograd from scratch**
  - A micrograd-style `Value` engine: recording the graph, topological sort, accumulating gradients
  - A neural network built on that engine
  - PyTorch autograd: `requires_grad`, `.backward()`, `.grad`, `grad_fn`
  - Gradient accumulation and zeroing; `no_grad` and `detach`
  - Gradient descent with raw tensors; custom `autograd.Function` and `gradcheck`
  - 🎥 **Recommended videos**
    - ⭐ [The spelled-out intro to neural networks and backpropagation: building micrograd](https://karpathy.ai/zero-to-hero.html) (Andrej Karpathy)
  - 📄 **Recommended reading**
    - ⭐ [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) (Baydin et al., 2018)
- [ ] **04 The PyTorch training loop**
  - `nn.Linear`, `nn.Module`, `nn.Sequential`, logits with `CrossEntropyLoss`
  - `Dataset`/`DataLoader`, torchvision MNIST, transforms and normalization, train/val/test splits
  - The training loop (`zero_grad → forward → loss → backward → step`), `train()` and `eval()` modes
  - Analysing mistakes; saving and loading a `state_dict`
  - 🎥 **Recommended videos**
    - ⭐ [makemore Part 2: MLP](https://karpathy.ai/zero-to-hero.html) (Andrej Karpathy)
  - 📄 **Recommended reading**
    - [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791) (LeCun et al., 1998)
    - [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747) (Xiao, Rasul & Vollgraf, 2017)
- [ ] **05 Optimization**
  - Narrow-valley loss surfaces: why the steepest direction limits the learning rate
  - SGD, momentum, RMSProp and Adam from scratch, verified against `torch.optim`
  - Comparing optimizers on a real network
  - The learning-rate range test; schedules (step, cosine, warmup, OneCycle)
  - Weight initialisation (Xavier/He, the zero-init symmetry problem); gradient clipping
  - 🎥 **Recommended videos**
    - ⭐ [Lecture 7: Training Neural Networks II (optimizers, learning rates)](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
    - [makemore Part 3: Activations & Gradients, BatchNorm](https://karpathy.ai/zero-to-hero.html) (Andrej Karpathy)
  - 📄 **Recommended reading**
    - ⭐ [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Kingma & Ba, 2014)
    - ⭐ [Understanding the Difficulty of Training Deep Feedforward Neural Networks](https://proceedings.mlr.press/v9/glorot10a.html) (Glorot & Bengio, 2010)
    - [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (He et al., 2015)
    - [On the Importance of Initialization and Momentum in Deep Learning](https://proceedings.mlr.press/v28/sutskever13.html) (Sutskever et al., 2013)
    - [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) (Smith, 2017)
    - [An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/abs/1609.04747) (Ruder, 2016)
- [ ] **06 Regularization and debugging**
  - Spotting overfitting on a small dataset
  - Dropout (inverted dropout from scratch), weight decay (L2 vs AdamW), early stopping
  - BatchNorm from scratch (running statistics, train vs eval); LayerNorm
  - Debugging checklist: initial loss ≈ ln K, overfitting one batch, finding NaNs, TensorBoard
  - 🎥 **Recommended videos**
    - ⭐ [makemore Part 3: Activations & Gradients, BatchNorm](https://karpathy.ai/zero-to-hero.html) (Andrej Karpathy)
    - [Lectures 6–7: Training Neural Networks I & II (BatchNorm, dropout, babysitting training)](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
  - 📄 **Recommended reading**
    - ⭐ [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (Srivastava et al., 2014)
    - ⭐ [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) (Ioffe & Szegedy, 2015)
    - ⭐ [A Recipe for Training Neural Networks (blog post)](http://karpathy.github.io/2019/04/25/recipe/) (Karpathy, 2019)
    - [Layer Normalization](https://arxiv.org/abs/1607.06450) (Ba, Kiros & Hinton, 2016)
    - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2017)
    - [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530) (Zhang et al., 2016)

</details>

### Phase 2: Computer vision 🚧

<details>
<summary>Modules 07–14: convolutions, CNNs, modern architectures, transfer learning, interpretability, detection, segmentation, ViT</summary>

- [ ] **07 Convolutions from scratch**
  - Kernels and feature detection (edges, blur), stride, padding, output-size formula
  - Multi-channel convolutions, pooling, receptive field
  - A NumPy `conv2d` verified against `nn.Conv2d`; parameter sharing vs fully connected layers
  - 🎥 **Recommended videos**
    - ⭐ [Lecture 5: Convolutional Neural Networks](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
  - 📄 **Recommended reading**
    - ⭐ [A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/abs/1603.07285) (Dumoulin & Visin, 2016)
- [ ] **08 CNNs in PyTorch**
  - The LeNet-style conv → pool → fully-connected pattern
  - CNNs on FashionMNIST and CIFAR-10, compared with MLPs
  - Data augmentation with `transforms.v2`
  - Visualising learned filters
  - 🎥 **Recommended videos**
    - ⭐ [Deep Computer Vision](https://introtodeeplearning.com/) (MIT 6.S191)
    - [Lecture 5: Convolutional Neural Networks](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
  - 📄 **Recommended reading**
    - ⭐ *ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)* (Krizhevsky, Sutskever & Hinton, 2012)
    - [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) (Zeiler & Fergus, 2013)
- [ ] **09 Modern CNN architectures**
  - VGG (stacking 3×3 convolutions), why plain deep networks degrade
  - Residual connections and ResNet blocks, built from scratch
  - A mini-ResNet on CIFAR-10; GlobalAvgPool; 1×1 convolutions
  - 🎥 **Recommended videos**
    - ⭐ [Lecture 9: CNN Architectures (AlexNet, VGG, GoogLeNet, ResNet)](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
  - 📄 **Recommended reading**
    - ⭐ [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385) (He et al., 2015)
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://arxiv.org/abs/1409.1556) (Simonyan & Zisserman, 2014)
    - [Network In Network](https://arxiv.org/abs/1312.4400) (Lin, Chen & Yan, 2013)
    - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (He et al., 2016)
- [ ] **10 Transfer learning**
  - Pretrained torchvision models and their weights and transforms
  - Feature extraction vs fine-tuning, freezing layers, different learning rates per layer
  - Fine-tuning on a custom dataset (Oxford-IIIT Pets)
  - 🎥 **Recommended videos**
    - ⭐ [Practical Deep Learning for Coders, Part 1 (lessons 1–2)](https://course.fast.ai/) (fast.ai)
  - 📄 **Recommended reading**
    - ⭐ [How Transferable are Features in Deep Neural Networks?](https://arxiv.org/abs/1411.1792) (Yosinski et al., 2014)
    - [Do Better ImageNet Models Transfer Better?](https://arxiv.org/abs/1805.08974) (Kornblith, Shlens & Le, 2018)
- [ ] **11 Interpreting CNNs**
  - Filter and feature-map visualization
  - Saliency maps from input gradients
  - Grad-CAM from scratch with forward/backward hooks
  - 🎥 **Recommended videos**
    - ⭐ [Lecture 12: Visualizing and Understanding](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
  - 📄 **Recommended reading**
    - ⭐ [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) (Selvaraju et al., 2016)
    - [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034) (Simonyan, Vedaldi & Zisserman, 2013)
    - [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) (Adebayo et al., 2018)
- [ ] **12 Object detection** _(Colab recommended)_
  - Bounding boxes, IoU and non-maximum suppression from scratch
  - Anchors and how two-stage detectors (Faster R-CNN) work
  - Fine-tuning torchvision Faster R-CNN on PennFudan; the mAP metric
  - 🎥 **Recommended videos**
    - ⭐ [Lecture 11: Detection and Segmentation](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
  - 📄 **Recommended reading**
    - ⭐ [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) (Ren et al., 2015)
    - [You Only Look Once: Unified, Real-Time Object Detection (YOLO)](https://arxiv.org/abs/1506.02640) (Redmon et al., 2015)
    - [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) (Lin et al., 2016)
    - [Focal Loss for Dense Object Detection (RetinaNet)](https://arxiv.org/abs/1708.02002) (Lin et al., 2017)
    - [End-to-End Object Detection with Transformers (DETR)](https://arxiv.org/abs/2005.12872) (Carion et al., 2020)
- [ ] **13 Segmentation with U-Net**
  - Semantic vs instance segmentation; per-pixel losses (cross-entropy, Dice)
  - Transposed convolutions and upsampling
  - U-Net from scratch on Oxford Pets masks (reused later for diffusion)
  - 🎥 **Recommended videos**
    - ⭐ [Lecture 11: Detection and Segmentation](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
  - 📄 **Recommended reading**
    - ⭐ [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger, Fischer & Brox, 2015)
    - [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) (Long, Shelhamer & Darrell, 2014)
    - [V-Net (Dice loss)](https://arxiv.org/abs/1606.04797) (Milletari, Navab & Ahmadi, 2016)
    - [Mask R-CNN](https://arxiv.org/abs/1703.06870) (He et al., 2017)
- [ ] **14 Attention and Vision Transformers**
  - Scaled dot-product self-attention from scratch; multi-head attention
  - Positional embeddings, Transformer blocks (LayerNorm, MLP, residuals)
  - Patch embeddings and a tiny ViT on CIFAR-10, compared with a CNN
  - 🎥 **Recommended videos**
    - ⭐ [Transformers, the tech behind LLMs (Chapter 5)](https://www.3blue1brown.com/lessons/gpt) (3Blue1Brown)
    - ⭐ [Attention in transformers, step-by-step (Chapter 6)](https://www.3blue1brown.com/lessons/attention) (3Blue1Brown)
  - 📄 **Recommended reading**
    - ⭐ [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
    - ⭐ [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020)
    - [Training Data-Efficient Image Transformers (DeiT)](https://arxiv.org/abs/2012.12877) (Touvron et al., 2020)

</details>

### Phase 3: Generative models 🚧

<details>
<summary>Modules 15–20: autoencoders, VAEs, GANs, advanced GANs, diffusion, latent diffusion</summary>

- [ ] **15 Autoencoders**
  - Encoder–bottleneck–decoder; undercomplete and convolutional autoencoders
  - Denoising autoencoders; visualizing the latent space
  - Anomaly detection from reconstruction error
  - 🎥 **Recommended videos**
    - ⭐ [Deep Generative Modeling](https://introtodeeplearning.com/) (MIT 6.S191)
    - [Lecture 13: Generative Models](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
  - 📄 **Recommended reading**
    - ⭐ [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647) (Hinton & Salakhutdinov, 2006)
    - *Extracting and Composing Robust Features with Denoising Autoencoders* (Vincent et al., 2008)
- [ ] **16 Variational autoencoders**
  - Probabilistic latent variables; the reparameterization trick
  - The ELBO: reconstruction loss + KL divergence, derived step by step
  - Sampling new images, latent interpolation, β-VAE
  - 🎥 **Recommended videos**
    - ⭐ [CS236 lectures on latent variable models and VAEs (2023)](https://deepgenerativemodels.github.io/) (Stanford CS236)
    - [Lecture 13: Generative Models (VAE part)](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
  - 📄 **Recommended reading**
    - ⭐ [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Kingma & Welling, 2013)
    - ⭐ [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) (Doersch, 2016)
    - [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) (Higgins et al., 2017)
- [ ] **17 GANs** _(Colab recommended)_
  - The generator vs discriminator game and its loss functions
  - DCGAN on MNIST/FashionMNIST
  - Training instabilities, mode collapse and practical tricks
  - 🎥 **Recommended videos**
    - ⭐ [CS236 lectures on GANs (2023)](https://deepgenerativemodels.github.io/) (Stanford CS236)
    - [Lecture 13: Generative Models (GAN part)](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
  - 📄 **Recommended reading**
    - ⭐ [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (Goodfellow et al., 2014)
    - ⭐ [Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN)](https://arxiv.org/abs/1511.06434) (Radford, Metz & Chintala, 2015)
    - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) (Salimans et al., 2016)
- [ ] **18 Advanced GANs** _(Colab recommended)_
  - Conditional GANs (class-controlled generation)
  - The Wasserstein distance and WGAN-GP (gradient penalty)
  - Evaluating generative models: an introduction to FID
  - 🎥 **Recommended videos**
    - ⭐ [CS236 lectures on GANs and evaluating generative models (2023)](https://deepgenerativemodels.github.io/) (Stanford CS236)
  - 📄 **Recommended reading**
    - ⭐ [Improved Training of Wasserstein GANs (WGAN-GP)](https://arxiv.org/abs/1704.00028) (Gulrajani et al., 2017)
    - [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) (Mirza & Osindero, 2014)
    - [Wasserstein GAN](https://arxiv.org/abs/1701.07875) (Arjovsky, Chintala & Bottou, 2017)
    - [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) (Heusel et al., 2017)
    - [A Style-Based Generator Architecture for GANs (StyleGAN)](https://arxiv.org/abs/1812.04948) (Karras, Laine & Aila, 2018)
- [ ] **19 Diffusion from scratch** _(Colab recommended)_
  - Forward noising, noise schedules, the closed-form q(xₜ | x₀)
  - The DDPM objective (predicting the noise) and the sampling loop
  - A U-Net with time embeddings and attention (reusing modules 13–14), trained on FashionMNIST
  - 🎥 **Recommended videos**
    - ⭐ [From Deep Learning Foundations to Stable Diffusion (Part 2)](https://course.fast.ai/Lessons/part2.html) (fast.ai)
    - [Diffusion Models Course (units 1–2)](https://huggingface.co/learn/diffusion-course) (Hugging Face)
    - [CS236 lectures on score-based and diffusion models (2023)](https://deepgenerativemodels.github.io/) (Stanford CS236)
  - 📄 **Recommended reading**
    - ⭐ [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) (Ho, Jain & Abbeel, 2020)
    - [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) (Sohl-Dickstein et al., 2015)
    - [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
    - [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) (Song et al., 2020)
- [ ] **20 Guided and latent diffusion** _(Colab recommended)_
  - Class-conditional diffusion and classifier-free guidance
  - Faster sampling with DDIM
  - Latent diffusion: VAE latents + text conditioning (how Stable Diffusion works)
  - Stable Diffusion with `diffusers`: text-to-image, img2img, inpainting
  - 🎥 **Recommended videos**
    - ⭐ [Lesson 9: Stable Diffusion](https://course.fast.ai/Lessons/lesson9.html) (fast.ai)
    - [Diffusion Models Course (guidance, Stable Diffusion units)](https://huggingface.co/learn/diffusion-course) (Hugging Face)
  - 📄 **Recommended reading**
    - ⭐ [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](https://arxiv.org/abs/2112.10752) (Rombach et al., 2021)
    - ⭐ [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (Ho & Salimans, 2022)
    - [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) (Song, Meng & Ermon, 2020)
    - [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) (Dhariwal & Nichol, 2021)
    - [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)

</details>

### Phase 4: NLP and large language models 🚧

<details>
<summary>Modules 21–28: word embeddings, RNNs, seq2seq, Transformers, GPT from scratch, Hugging Face, LoRA fine-tuning, RAG</summary>

- [ ] **21 Text representation and word embeddings**
  - Tokenization basics (characters, words, subwords), vocabularies, padding and batching text
  - From one-hot and bag-of-words to dense embeddings; `nn.Embedding`
  - Word2vec (skip-gram with negative sampling) from scratch; similarity and analogies
  - 🎥 **Recommended videos**
    - ⭐ [Word vectors lectures (word2vec, GloVe)](https://web.stanford.edu/class/cs224n/) (Stanford CS224n)
    - [makemore Part 2: MLP (character embeddings)](https://karpathy.ai/zero-to-hero.html) (Andrej Karpathy)
  - 📄 **Recommended reading**
    - ⭐ [Efficient Estimation of Word Representations in Vector Space (word2vec)](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013)
    - ⭐ [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (Mikolov et al., 2013)
    - [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) (Pennington, Socher & Manning, 2014)
- [ ] **22 RNNs, LSTMs and GRUs**
  - Recurrent networks from scratch; backpropagation through time
  - Vanishing and exploding gradients in sequences; LSTM and GRU gates
  - A character-level language model; text classification with an LSTM
  - 🎥 **Recommended videos**
    - ⭐ [Recurrent neural networks and LSTMs lectures](https://web.stanford.edu/class/cs224n/) (Stanford CS224n)
    - [Deep Sequence Modeling](https://introtodeeplearning.com/) (MIT 6.S191)
    - [Lecture 10: Recurrent Neural Networks](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n (2017))
  - 📄 **Recommended reading**
    - ⭐ [The Unreasonable Effectiveness of Recurrent Neural Networks (blog post)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Karpathy, 2015)
    - [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) (Hochreiter & Schmidhuber, 1997)
    - [Learning Long-Term Dependencies with Gradient Descent is Difficult](https://doi.org/10.1109/72.279181) (Bengio, Simard & Frasconi, 1994)
    - [Learning Phrase Representations using RNN Encoder–Decoder (GRU)](https://arxiv.org/abs/1406.1078) (Cho et al., 2014)
- [ ] **23 Seq2seq and attention**
  - Encoder–decoder models, teacher forcing, greedy and beam search decoding
  - The bottleneck problem and Bahdanau (additive) attention, visualised as alignment maps
  - A toy translation task (e.g. date-format conversion), trained offline
  - 🎥 **Recommended videos**
    - ⭐ [Machine translation, seq2seq and attention lecture](https://web.stanford.edu/class/cs224n/) (Stanford CS224n)
  - 📄 **Recommended reading**
    - ⭐ [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) (Sutskever, Vinyals & Le, 2014)
    - ⭐ [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau, Cho & Bengio, 2014)
    - [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) (Luong, Pham & Manning, 2015)
- [ ] **24 Transformers for language**
  - Encoder vs decoder Transformers; causal masking and padding masks
  - Positional encodings: sinusoidal, learned, RoPE
  - A Transformer encoder for text classification, built on module 14's attention
  - 🎥 **Recommended videos**
    - ⭐ [Self-attention and Transformers lecture](https://web.stanford.edu/class/cs224n/) (Stanford CS224n)
    - [Attention in transformers, step-by-step (Chapter 6)](https://www.3blue1brown.com/lessons/attention) (3Blue1Brown)
  - 📄 **Recommended reading**
    - ⭐ [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
    - ⭐ [The Annotated Transformer (blog post)](https://nlp.seas.harvard.edu/annotated-transformer/) (Harvard NLP)
    - [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
- [ ] **25 GPT from scratch** _(Colab recommended)_
  - Byte-pair encoding (BPE) tokenizer from scratch
  - A small decoder-only GPT trained on Shakespeare
  - Sampling: temperature, top-k, top-p; the KV cache for fast generation; scaling laws in brief
  - 🎥 **Recommended videos**
    - ⭐ [Let's build GPT: from scratch, in code, spelled out](https://karpathy.ai/zero-to-hero.html) (Andrej Karpathy)
    - ⭐ [Let's build the GPT Tokenizer](https://karpathy.ai/zero-to-hero.html) (Andrej Karpathy)
    - [Let's reproduce GPT-2 (124M)](https://karpathy.ai/zero-to-hero.html) (Andrej Karpathy)
    - [How might LLMs store facts (Chapter 7)](https://www.3blue1brown.com/lessons/mlp) (3Blue1Brown)
  - 📄 **Recommended reading**
    - ⭐ [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019)
    - ⭐ [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) (Sennrich, Haddow & Birch, 2015)
    - [Improving Language Understanding by Generative Pre-Training (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al., 2018)
    - [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) (Brown et al., 2020)
    - [The Curious Case of Neural Text Degeneration (top-p sampling)](https://arxiv.org/abs/1904.09751) (Holtzman et al., 2019)
    - [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020)
    - [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022)
- [ ] **26 Pretrained Transformers with Hugging Face**
  - The `transformers` library: tokenizers, models, pipelines
  - BERT-style vs GPT-style models and what each is good at
  - Fine-tuning a pretrained encoder for text classification; evaluation
  - 🎥 **Recommended videos**
    - ⭐ [LLM Course (chapters on Transformers, tokenizers and fine-tuning)](https://huggingface.co/learn/llm-course) (Hugging Face)
    - [Pretraining lecture](https://web.stanford.edu/class/cs224n/) (Stanford CS224n)
  - 📄 **Recommended reading**
    - ⭐ [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)
    - [HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/abs/1910.03771) (Wolf et al., 2019)
    - [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) (Liu et al., 2019)
    - [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019)
    - [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](https://arxiv.org/abs/1910.10683) (Raffel et al., 2019)
- [ ] **27 Fine-tuning LLMs and LoRA** _(Colab recommended)_
  - How LLMs are trained: pretraining → instruction tuning → preference alignment (RLHF, DPO concepts)
  - LoRA from scratch, then with the `peft` library
  - Instruction-tuning a small open model; chat templates; memory-saving tricks (mixed precision, quantization basics)
  - 🎥 **Recommended videos**
    - ⭐ [Deep Dive into LLMs like ChatGPT](https://youtu.be/7xTGNNLPyMI) (Andrej Karpathy)
    - [Prompting, instruction fine-tuning and RLHF lecture](https://web.stanford.edu/class/cs224n/) (Stanford CS224n)
    - [LLM Course (fine-tuning chapters)](https://huggingface.co/learn/llm-course) (Hugging Face)
  - 📄 **Recommended reading**
    - ⭐ [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
    - ⭐ [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022)
    - [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)
    - [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
    - [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) (Wei et al., 2021)
    - [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (Micikevicius et al., 2017)
- [ ] **28 Embeddings, retrieval and RAG**
  - Sentence embeddings and semantic search; vector similarity at scale
  - A retrieval-augmented generation (RAG) pipeline from scratch: chunking, retrieval, prompting
  - Evaluating LLM outputs; limitations (hallucination, context length) and where to go next
  - 🎥 **Recommended videos**
    - ⭐ [LLM Course (semantic search and embeddings sections)](https://huggingface.co/learn/llm-course) (Hugging Face)
    - [Large Language Models explained briefly](https://www.3blue1brown.com/lessons/mini-llm) (3Blue1Brown)
  - 📄 **Recommended reading**
    - ⭐ [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
    - ⭐ [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)
    - [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020)
    - [Billion-scale Similarity Search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734) (Johnson, Douze & Jégou, 2017)
    - [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (Liu et al., 2023)
    - [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) (Ji et al., 2022)

</details>

### Phase 5: Multimodal AI 🚧

<details>
<summary>Modules 29–32: CLIP, image captioning, vision-language models, audio and speech</summary>

- [ ] **29 Contrastive image–text learning (CLIP)**
  - Shared embedding spaces: why put images and text in the same vector space
  - The symmetric contrastive (InfoNCE) loss from scratch; temperature and batch size
  - Train a mini-CLIP (small image encoder + text encoder) on a small captioned dataset
  - Zero-shot classification and text↔image retrieval with a pretrained CLIP (`open_clip` / Hugging Face)
  - Prompt templates and prompt ensembling; failure modes (bias, typographic attacks)
  - 🎥 **Recommended videos**
    - ⭐ [Community Computer Vision Course: CLIP and relatives](https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/clip-and-relatives/clip) (Hugging Face)
    - [Vision and Language lecture (slides)](https://cs231n.stanford.edu/schedule.html) (Stanford CS231n)
  - 📄 **Recommended reading**
    - ⭐ [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
    - ⭐ [Representation Learning with Contrastive Predictive Coding (InfoNCE)](https://arxiv.org/abs/1807.03748) (van den Oord, Li & Vinyals, 2018)
    - [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN)](https://arxiv.org/abs/2102.05918) (Jia et al., 2021)
    - [Sigmoid Loss for Language Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343) (Zhai et al., 2023)
    - [Reproducible Scaling Laws for Contrastive Language-Image Learning (OpenCLIP)](https://arxiv.org/abs/2212.07143) (Cherti et al., 2022)
- [ ] **30 Image captioning** _(Colab recommended)_
  - Encoder–decoder captioning: a pretrained vision encoder plus a Transformer decoder
  - Cross-attention from text tokens to image patches; visualizing where the model looks
  - Teacher forcing for training; greedy decoding and beam search for generation
  - Evaluating with BLEU and CIDEr, and why those metrics fall short
  - BLIP as a pretrained reference to compare against
  - 🎥 **Recommended videos**
    - ⭐ [Recurrent Neural Networks lecture (image captioning and soft attention)](https://cs231n.stanford.edu/2017/syllabus.html) (Stanford CS231n)
    - [Community Computer Vision Course: multimodal tasks and models](https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/tasks-models-part1) (Hugging Face)
  - 📄 **Recommended reading**
    - ⭐ [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) (Vinyals et al., 2014)
    - ⭐ [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044) (Xu et al., 2015)
    - [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086) (Li et al., 2022)
    - [CIDEr: Consensus-based Image Description Evaluation](https://arxiv.org/abs/1411.5726) (Vedantam, Zitnick & Parikh, 2014)
- [ ] **31 Vision-language models** _(Colab recommended)_
  - LLaVA-style architecture: frozen vision encoder → projector → LLM; image tokens in the LLM's context
  - Build and train the projector on a small image–text set
  - Visual instruction tuning with LoRA (builds on module 27)
  - Visual question answering with a pretrained small VLM
  - Evaluating VLMs; object hallucination and how to measure it
  - 🎥 **Recommended videos**
    - ⭐ [Community Computer Vision Course: introduction to vision language models](https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/vlm-intro) (Hugging Face)
    - [Multimodality guest lecture](https://web.stanford.edu/class/cs224n/) (Stanford CS224n)
  - 📄 **Recommended reading**
    - ⭐ [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) (Liu et al., 2023)
    - ⭐ [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) (Alayrac et al., 2022)
    - [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs](https://arxiv.org/abs/2301.12597) (Li et al., 2023)
    - [Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](https://arxiv.org/abs/2310.03744) (Liu et al., 2023)
    - [Evaluating Object Hallucination in Large Vision-Language Models (POPE)](https://arxiv.org/abs/2305.10355) (Li et al., 2023)
- [ ] **32 Audio and speech**
  - Waveforms and sampling; the STFT and (log-)mel spectrograms
  - A small CNN audio classifier on spectrograms; SpecAugment
  - Speech recognition with Whisper (an encoder–decoder over spectrograms); CTC in brief
  - Audio–text contrastive learning (CLAP): CLIP for sound
  - 🎥 **Recommended videos**
    - ⭐ [Audio Course (chapter 1: working with audio data)](https://huggingface.co/learn/audio-course/chapter1/audio_data) (Hugging Face)
    - [Audio Course (chapter 5: automatic speech recognition)](https://huggingface.co/learn/audio-course/chapter5/introduction) (Hugging Face)
  - 📄 **Recommended reading**
    - ⭐ [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356) (Radford et al., 2022)
    - ⭐ [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (Graves et al., 2006)
    - [wav2vec 2.0: Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) (Baevski et al., 2020)
    - [CLAP: Learning Audio Concepts From Natural Language Supervision](https://arxiv.org/abs/2206.04769) (Elizalde et al., 2022)
    - [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779) (Park et al., 2019)

</details>

### Phase 6: Capstone 🚧

<details>
<summary>Module 33: pick a vision, generative, language or multimodal project</summary>

- [ ] **33 Capstone project** (pick one)
  - (a) Vision: fine-tune and evaluate a classifier or segmenter on your own dataset, with Grad-CAM analysis
  - (b) Generative: train a conditional diffusion model or VAE on a small dataset
  - (c) Language: fine-tune a small LLM with LoRA for a task of your choice, or build and evaluate a RAG system
  - (d) Multimodal: build an image–text or audio–text retrieval system, or fine-tune a small VLM for a task of your choice
  - Guided checklist: data → baseline → improvements → evaluation → write-up, plus a reference solution
  - 📄 **Recommended reading**
    - ⭐ [A Recipe for Training Neural Networks (blog post)](http://karpathy.github.io/2019/04/25/recipe/) (Karpathy, 2019)

</details>

## For maintainers: rebuilding notebooks

Notebooks are generated from the sources in `tools/src/`:

```bash
python tools/build.py 03            # rebuild module 03
python tools/build.py all --execute # rebuild everything and run lesson + solutions quickly (FAST_DEV_RUN=1)
```

Recommended papers and videos live in `tools/papers.py`. After editing it, run `python tools/update_readme.py` and rebuild the lessons.

Most video links go to the official course or lesson page, which links to the video. Video courses used:
[3Blue1Brown](https://www.3blue1brown.com/lessons/gradient-descent) ·
[Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) ·
[Stanford CS231n (2017)](https://cs231n.stanford.edu/2017/syllabus.html) ·
[Stanford CS224n](https://web.stanford.edu/class/cs224n/) ·
[Stanford CS236](https://deepgenerativemodels.github.io/) ·
[MIT 6.S191](https://introtodeeplearning.com/) ·
[fast.ai](https://course.fast.ai/) ·
Hugging Face [LLM](https://huggingface.co/learn/llm-course), [diffusion](https://huggingface.co/learn/diffusion-course),
[computer vision](https://huggingface.co/learn/computer-vision-course) and [audio](https://huggingface.co/learn/audio-course) courses.
