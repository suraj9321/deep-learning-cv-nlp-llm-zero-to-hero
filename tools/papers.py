"""Recommended reading per module: the single source for README.md and the lesson notebooks.

Each entry: (title, authors_year, url, why, essential)
- essential=True  -> marked ⭐ ("read this one"); the others are optional deeper reading.
Modules without papers (or not needing them) are simply absent.
"""

def arxiv(id_):
    return f"https://arxiv.org/abs/{id_}"


PAPERS = {
    "00": [
        ("PyTorch: An Imperative Style, High-Performance Deep Learning Library", "Paszke et al., 2019",
         arxiv("1912.01703"), "How PyTorch's design (eager execution, autograd, devices) came about.", False),
    ],
    "01": [
        ("The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain",
         "Rosenblatt, 1958", "https://doi.org/10.1037/h0042519", "The original perceptron; read it for history.", False),
        ("Multilayer Feedforward Networks are Universal Approximators", "Hornik, Stinchcombe & White, 1989",
         "https://doi.org/10.1016/0893-6080(89)90020-8", "Why one hidden layer can, in principle, fit any function.", False),
        ("Rectified Linear Units Improve Restricted Boltzmann Machines", "Nair & Hinton, 2010",
         "", "Where ReLU entered mainstream deep learning.", False),
    ],
    "02": [
        ("Learning Representations by Back-propagating Errors", "Rumelhart, Hinton & Williams, 1986",
         "https://doi.org/10.1038/323533a0", "The classic 4-page paper that popularised backprop.", True),
    ],
    "03": [
        ("Automatic Differentiation in Machine Learning: a Survey", "Baydin et al., 2018",
         arxiv("1502.05767"), "Forward vs reverse mode AD; what autograd engines actually do.", True),
    ],
    "04": [
        ("Gradient-Based Learning Applied to Document Recognition", "LeCun et al., 1998",
         "https://doi.org/10.1109/5.726791", "The MNIST / LeNet paper. Long; sections I–II are enough for now.", False),
        ("Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms", "Xiao, Rasul & Vollgraf, 2017",
         arxiv("1708.07747"), "Why FashionMNIST was created as a harder drop-in for MNIST.", False),
    ],
    "05": [
        ("Adam: A Method for Stochastic Optimization", "Kingma & Ba, 2014",
         arxiv("1412.6980"), "The Adam update rule and bias correction you implemented.", True),
        ("Understanding the Difficulty of Training Deep Feedforward Neural Networks", "Glorot & Bengio, 2010",
         "https://proceedings.mlr.press/v9/glorot10a.html", "Xavier initialisation and activation statistics.", True),
        ("Delving Deep into Rectifiers", "He et al., 2015",
         arxiv("1502.01852"), "He (Kaiming) initialisation for ReLU networks.", False),
        ("On the Importance of Initialization and Momentum in Deep Learning", "Sutskever et al., 2013",
         "https://proceedings.mlr.press/v28/sutskever13.html", "Momentum and Nesterov momentum in practice.", False),
        ("Cyclical Learning Rates for Training Neural Networks", "Smith, 2017",
         arxiv("1506.01186"), "The LR range test.", False),
        ("An Overview of Gradient Descent Optimization Algorithms", "Ruder, 2016",
         arxiv("1609.04747"), "A readable survey of every optimizer in this module.", False),
    ],
    "06": [
        ("Dropout: A Simple Way to Prevent Neural Networks from Overfitting", "Srivastava et al., 2014",
         "https://jmlr.org/papers/v15/srivastava14a.html", "The dropout paper; sections 1–4 and 7.", True),
        ("Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
         "Ioffe & Szegedy, 2015", arxiv("1502.03167"), "BatchNorm, including the train/eval statistics.", True),
        ("Layer Normalization", "Ba, Kiros & Hinton, 2016", arxiv("1607.06450"), "The normalization Transformers use.", False),
        ("Decoupled Weight Decay Regularization", "Loshchilov & Hutter, 2017",
         arxiv("1711.05101"), "Why AdamW differs from Adam + L2.", False),
        ("Understanding Deep Learning Requires Rethinking Generalization", "Zhang et al., 2016",
         arxiv("1611.03530"), "Networks can memorise random labels; what that means for regularization.", False),
        ("A Recipe for Training Neural Networks (blog post)", "Karpathy, 2019",
         "http://karpathy.github.io/2019/04/25/recipe/", "The debugging mindset from section 6, in depth.", True),
    ],
    "07": [
        ("A Guide to Convolution Arithmetic for Deep Learning", "Dumoulin & Visin, 2016",
         arxiv("1603.07285"), "Stride, padding and output sizes, with diagrams.", True),
    ],
    "08": [
        ("ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)", "Krizhevsky, Sutskever & Hinton, 2012",
         "", "The paper that started the deep learning boom in vision.", True),
        ("Visualizing and Understanding Convolutional Networks", "Zeiler & Fergus, 2013",
         arxiv("1311.2901"), "What CNN filters learn, layer by layer.", False),
    ],
    "09": [
        ("Deep Residual Learning for Image Recognition (ResNet)", "He et al., 2015",
         arxiv("1512.03385"), "Residual connections, one of the most important ideas in deep learning.", True),
        ("Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)", "Simonyan & Zisserman, 2014",
         arxiv("1409.1556"), "Stacks of 3×3 convolutions.", False),
        ("Network In Network", "Lin, Chen & Yan, 2013", arxiv("1312.4400"), "1×1 convolutions and global average pooling.", False),
        ("Identity Mappings in Deep Residual Networks", "He et al., 2016", arxiv("1603.05027"), "Pre-activation ResNets.", False),
    ],
    "10": [
        ("How Transferable are Features in Deep Neural Networks?", "Yosinski et al., 2014",
         arxiv("1411.1792"), "Which layers transfer between tasks, and why.", True),
        ("Do Better ImageNet Models Transfer Better?", "Kornblith, Shlens & Le, 2018",
         arxiv("1805.08974"), "Empirical guidance on choosing a backbone.", False),
    ],
    "11": [
        ("Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", "Selvaraju et al., 2016",
         arxiv("1610.02391"), "The method you implement in this module.", True),
        ("Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps",
         "Simonyan, Vedaldi & Zisserman, 2013", arxiv("1312.6034"), "Gradient saliency maps.", False),
        ("Sanity Checks for Saliency Maps", "Adebayo et al., 2018",
         arxiv("1810.03292"), "Why you shouldn't trust every pretty heatmap.", False),
    ],
    "12": [
        ("Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", "Ren et al., 2015",
         arxiv("1506.01497"), "The detector you fine-tune; anchors and RPNs.", True),
        ("You Only Look Once: Unified, Real-Time Object Detection (YOLO)", "Redmon et al., 2015",
         arxiv("1506.02640"), "The one-stage alternative.", False),
        ("Feature Pyramid Networks for Object Detection", "Lin et al., 2016", arxiv("1612.03144"), "Multi-scale features.", False),
        ("Focal Loss for Dense Object Detection (RetinaNet)", "Lin et al., 2017", arxiv("1708.02002"), "Handling class imbalance.", False),
        ("End-to-End Object Detection with Transformers (DETR)", "Carion et al., 2020",
         arxiv("2005.12872"), "Detection without anchors or NMS (read after module 14).", False),
    ],
    "13": [
        ("U-Net: Convolutional Networks for Biomedical Image Segmentation", "Ronneberger, Fischer & Brox, 2015",
         arxiv("1505.04597"), "The architecture you build, and later reuse for diffusion.", True),
        ("Fully Convolutional Networks for Semantic Segmentation", "Long, Shelhamer & Darrell, 2014",
         arxiv("1411.4038"), "The first end-to-end segmentation network.", False),
        ("V-Net (Dice loss)", "Milletari, Navab & Ahmadi, 2016", arxiv("1606.04797"), "Where the Dice loss comes from.", False),
        ("Mask R-CNN", "He et al., 2017", arxiv("1703.06870"), "Instance segmentation.", False),
    ],
    "14": [
        ("Attention Is All You Need", "Vaswani et al., 2017",
         arxiv("1706.03762"), "The Transformer. Read section 3 closely.", True),
        ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)", "Dosovitskiy et al., 2020",
         arxiv("2010.11929"), "The Vision Transformer you build.", True),
        ("Training Data-Efficient Image Transformers (DeiT)", "Touvron et al., 2020",
         arxiv("2012.12877"), "Making ViTs work without huge datasets.", False),
    ],
    "15": [
        ("Reducing the Dimensionality of Data with Neural Networks", "Hinton & Salakhutdinov, 2006",
         "https://doi.org/10.1126/science.1127647", "Deep autoencoders vs PCA.", True),
        ("Extracting and Composing Robust Features with Denoising Autoencoders", "Vincent et al., 2008",
         "", "Denoising autoencoders.", False),
    ],
    "16": [
        ("Auto-Encoding Variational Bayes", "Kingma & Welling, 2013",
         arxiv("1312.6114"), "The VAE paper; the reparameterization trick.", True),
        ("Tutorial on Variational Autoencoders", "Doersch, 2016",
         arxiv("1606.05908"), "A gentler derivation of the ELBO. Read this first.", True),
        ("beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework", "Higgins et al., 2017",
         "https://openreview.net/forum?id=Sy2fzU9gl", "Disentangled latents via β.", False),
    ],
    "17": [
        ("Generative Adversarial Networks", "Goodfellow et al., 2014", arxiv("1406.2661"), "The original GAN game.", True),
        ("Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN)", "Radford, Metz & Chintala, 2015",
         arxiv("1511.06434"), "The architecture guidelines you follow.", True),
        ("Improved Techniques for Training GANs", "Salimans et al., 2016", arxiv("1606.03498"), "Practical stabilisation tricks.", False),
    ],
    "18": [
        ("Conditional Generative Adversarial Nets", "Mirza & Osindero, 2014", arxiv("1411.1784"), "Class-conditional GANs.", False),
        ("Wasserstein GAN", "Arjovsky, Chintala & Bottou, 2017", arxiv("1701.07875"), "Why the Wasserstein distance helps.", False),
        ("Improved Training of Wasserstein GANs (WGAN-GP)", "Gulrajani et al., 2017",
         arxiv("1704.00028"), "The gradient penalty you implement.", True),
        ("GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)", "Heusel et al., 2017",
         arxiv("1706.08500"), "Introduces the FID metric.", False),
        ("A Style-Based Generator Architecture for GANs (StyleGAN)", "Karras, Laine & Aila, 2018",
         arxiv("1812.04948"), "State-of-the-art GAN design, for further reading.", False),
    ],
    "19": [
        ("Denoising Diffusion Probabilistic Models (DDPM)", "Ho, Jain & Abbeel, 2020",
         arxiv("2006.11239"), "The paper you implement.", True),
        ("Deep Unsupervised Learning using Nonequilibrium Thermodynamics", "Sohl-Dickstein et al., 2015",
         arxiv("1503.03585"), "Where diffusion models started.", False),
        ("Improved Denoising Diffusion Probabilistic Models", "Nichol & Dhariwal, 2021",
         arxiv("2102.09672"), "Cosine schedule and learned variances.", False),
        ("Score-Based Generative Modeling through Stochastic Differential Equations", "Song et al., 2020",
         arxiv("2011.13456"), "The continuous-time view (advanced).", False),
    ],
    "20": [
        ("High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)", "Rombach et al., 2021",
         arxiv("2112.10752"), "How Stable Diffusion works.", True),
        ("Classifier-Free Diffusion Guidance", "Ho & Salimans, 2022", arxiv("2207.12598"), "The guidance you implement.", True),
        ("Denoising Diffusion Implicit Models (DDIM)", "Song, Meng & Ermon, 2020", arxiv("2010.02502"), "Faster sampling.", False),
        ("Diffusion Models Beat GANs on Image Synthesis", "Dhariwal & Nichol, 2021", arxiv("2105.05233"), "Classifier guidance.", False),
        ("Learning Transferable Visual Models From Natural Language Supervision (CLIP)", "Radford et al., 2021",
         arxiv("2103.00020"), "The text encoder behind text-to-image models.", False),
    ],
    "21": [
        ("Efficient Estimation of Word Representations in Vector Space (word2vec)", "Mikolov et al., 2013",
         arxiv("1301.3781"), "Skip-gram and CBOW.", True),
        ("Distributed Representations of Words and Phrases and their Compositionality", "Mikolov et al., 2013",
         arxiv("1310.4546"), "Negative sampling, which you implement.", True),
        ("GloVe: Global Vectors for Word Representation", "Pennington, Socher & Manning, 2014",
         "https://nlp.stanford.edu/pubs/glove.pdf", "Count-based embeddings.", False),
    ],
    "22": [
        ("Long Short-Term Memory", "Hochreiter & Schmidhuber, 1997",
         "https://doi.org/10.1162/neco.1997.9.8.1735", "The LSTM paper (dense; skim the cell design).", False),
        ("Learning Long-Term Dependencies with Gradient Descent is Difficult", "Bengio, Simard & Frasconi, 1994",
         "https://doi.org/10.1109/72.279181", "Why plain RNNs forget.", False),
        ("Learning Phrase Representations using RNN Encoder–Decoder (GRU)", "Cho et al., 2014", arxiv("1406.1078"), "The GRU.", False),
        ("The Unreasonable Effectiveness of Recurrent Neural Networks (blog post)", "Karpathy, 2015",
         "http://karpathy.github.io/2015/05/21/rnn-effectiveness/", "Character-level language models, intuitively.", True),
    ],
    "23": [
        ("Sequence to Sequence Learning with Neural Networks", "Sutskever, Vinyals & Le, 2014",
         arxiv("1409.3215"), "The encoder–decoder setup.", True),
        ("Neural Machine Translation by Jointly Learning to Align and Translate", "Bahdanau, Cho & Bengio, 2014",
         arxiv("1409.0473"), "The attention mechanism you implement.", True),
        ("Effective Approaches to Attention-based Neural Machine Translation", "Luong, Pham & Manning, 2015",
         arxiv("1508.04025"), "Multiplicative attention variants.", False),
    ],
    "24": [
        ("Attention Is All You Need", "Vaswani et al., 2017", arxiv("1706.03762"), "Reread it now with language in mind.", True),
        ("The Annotated Transformer (blog post)", "Harvard NLP",
         "https://nlp.seas.harvard.edu/annotated-transformer/", "The paper as runnable code, line by line.", True),
        ("RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)", "Su et al., 2021",
         arxiv("2104.09864"), "The positional encoding used in most modern LLMs.", False),
    ],
    "25": [
        ("Language Models are Unsupervised Multitask Learners (GPT-2)", "Radford et al., 2019",
         "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
         "The architecture you build.", True),
        ("Neural Machine Translation of Rare Words with Subword Units (BPE)", "Sennrich, Haddow & Birch, 2015",
         arxiv("1508.07909"), "The tokenizer you implement.", True),
        ("Improving Language Understanding by Generative Pre-Training (GPT-1)", "Radford et al., 2018",
         "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf", "Pretrain-then-fine-tune.", False),
        ("Language Models are Few-Shot Learners (GPT-3)", "Brown et al., 2020", arxiv("2005.14165"), "In-context learning at scale.", False),
        ("The Curious Case of Neural Text Degeneration (top-p sampling)", "Holtzman et al., 2019",
         arxiv("1904.09751"), "Why sampling strategy matters.", False),
        ("Scaling Laws for Neural Language Models", "Kaplan et al., 2020", arxiv("2001.08361"), "How loss scales with size.", False),
        ("Training Compute-Optimal Large Language Models (Chinchilla)", "Hoffmann et al., 2022",
         arxiv("2203.15556"), "Balancing model size and data.", False),
    ],
    "26": [
        ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "Devlin et al., 2018",
         arxiv("1810.04805"), "Masked language modelling; the encoder you fine-tune.", True),
        ("HuggingFace's Transformers: State-of-the-art Natural Language Processing", "Wolf et al., 2019",
         arxiv("1910.03771"), "The library's design.", False),
        ("RoBERTa: A Robustly Optimized BERT Pretraining Approach", "Liu et al., 2019", arxiv("1907.11692"), "Training BERT better.", False),
        ("DistilBERT, a distilled version of BERT", "Sanh et al., 2019", arxiv("1910.01108"), "Knowledge distillation.", False),
        ("Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)", "Raffel et al., 2019",
         arxiv("1910.10683"), "Encoder–decoder pretraining.", False),
    ],
    "27": [
        ("LoRA: Low-Rank Adaptation of Large Language Models", "Hu et al., 2021", arxiv("2106.09685"), "The method you implement.", True),
        ("Training Language Models to Follow Instructions with Human Feedback (InstructGPT)", "Ouyang et al., 2022",
         arxiv("2203.02155"), "SFT → reward model → RLHF.", True),
        ("Direct Preference Optimization (DPO)", "Rafailov et al., 2023", arxiv("2305.18290"), "Alignment without RL.", False),
        ("QLoRA: Efficient Finetuning of Quantized LLMs", "Dettmers et al., 2023", arxiv("2305.14314"), "Fine-tuning on small GPUs.", False),
        ("Finetuned Language Models Are Zero-Shot Learners (FLAN)", "Wei et al., 2021", arxiv("2109.01652"), "Instruction tuning.", False),
        ("Mixed Precision Training", "Micikevicius et al., 2017", arxiv("1710.03740"), "float16 training and loss scaling.", False),
    ],
    "28": [
        ("Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "Lewis et al., 2020",
         arxiv("2005.11401"), "The original RAG paper.", True),
        ("Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", "Reimers & Gurevych, 2019",
         arxiv("1908.10084"), "The sentence-embedding models you use.", True),
        ("Dense Passage Retrieval for Open-Domain Question Answering", "Karpukhin et al., 2020", arxiv("2004.04906"), "Dense retrieval.", False),
        ("Billion-scale Similarity Search with GPUs (FAISS)", "Johnson, Douze & Jégou, 2017", arxiv("1702.08734"), "Vector search at scale.", False),
        ("Lost in the Middle: How Language Models Use Long Contexts", "Liu et al., 2023", arxiv("2307.03172"), "Context-length limitations.", False),
        ("Survey of Hallucination in Natural Language Generation", "Ji et al., 2022", arxiv("2202.03629"), "Why models make things up.", False),
    ],
    "29": [
        ("Learning Transferable Visual Models From Natural Language Supervision (CLIP)", "Radford et al., 2021",
         arxiv("2103.00020"), "The model you build a mini version of.", True),
        ("Representation Learning with Contrastive Predictive Coding (InfoNCE)", "van den Oord, Li & Vinyals, 2018",
         arxiv("1807.03748"), "Where the contrastive loss comes from.", True),
        ("Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN)",
         "Jia et al., 2021", arxiv("2102.05918"), "CLIP-style training on noisier, bigger data.", False),
        ("Sigmoid Loss for Language Image Pre-Training (SigLIP)", "Zhai et al., 2023", arxiv("2303.15343"),
         "A simpler loss that drops the softmax over the batch.", False),
        ("Reproducible Scaling Laws for Contrastive Language-Image Learning (OpenCLIP)", "Cherti et al., 2022",
         arxiv("2212.07143"), "The open models used in this module.", False),
    ],
    "30": [
        ("Show and Tell: A Neural Image Caption Generator", "Vinyals et al., 2014", arxiv("1411.4555"),
         "The original encoder–decoder captioner.", True),
        ("Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", "Xu et al., 2015",
         arxiv("1502.03044"), "Attending to image regions while generating.", True),
        ("BLIP: Bootstrapping Language-Image Pre-training", "Li et al., 2022", arxiv("2201.12086"),
         "The pretrained captioner used as a reference.", False),
        ("CIDEr: Consensus-based Image Description Evaluation", "Vedantam, Zitnick & Parikh, 2014", arxiv("1411.5726"),
         "The captioning metric, and what it rewards.", False),
    ],
    "31": [
        ("Visual Instruction Tuning (LLaVA)", "Liu et al., 2023", arxiv("2304.08485"),
         "Vision encoder → projector → LLM: the architecture you build.", True),
        ("Flamingo: a Visual Language Model for Few-Shot Learning", "Alayrac et al., 2022", arxiv("2204.14198"),
         "Feeding images to a frozen LLM with cross-attention.", True),
        ("BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs", "Li et al., 2023",
         arxiv("2301.12597"), "The Q-Former bridge between frozen models.", False),
        ("Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)", "Liu et al., 2023", arxiv("2310.03744"),
         "Simple changes that make LLaVA much better.", False),
        ("Evaluating Object Hallucination in Large Vision-Language Models (POPE)", "Li et al., 2023",
         arxiv("2305.10355"), "When VLMs describe objects that are not there.", False),
    ],
    "32": [
        ("Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)", "Radford et al., 2022",
         arxiv("2212.04356"), "The speech recognizer you use.", True),
        ("Connectionist Temporal Classification", "Graves et al., 2006", "https://www.cs.toronto.edu/~graves/icml_2006.pdf",
         "Training on audio without frame-level alignments.", True),
        ("wav2vec 2.0: Self-Supervised Learning of Speech Representations", "Baevski et al., 2020",
         arxiv("2006.11477"), "Self-supervised pretraining on raw audio.", False),
        ("CLAP: Learning Audio Concepts From Natural Language Supervision", "Elizalde et al., 2022",
         arxiv("2206.04769"), "CLIP for sound.", False),
        ("SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition", "Park et al., 2019",
         arxiv("1904.08779"), "Augmenting spectrograms.", False),
    ],
    "33": [
        ("A Recipe for Training Neural Networks (blog post)", "Karpathy, 2019",
         "http://karpathy.github.io/2019/04/25/recipe/", "Reread it before starting your project.", True),
    ],
}


# ---------------------------------------------------------------------------------------------------------------
# Recommended videos per module. Same tuple format: (title, source, url, why, essential).
# Links point to official course or lesson pages (verified 2026-11-24); each page links to the video itself.
# ---------------------------------------------------------------------------------------------------------------
_3B1B = "https://www.3blue1brown.com/lessons/"
_KARPATHY = "https://karpathy.ai/zero-to-hero.html"           # every Zero-to-Hero video is linked from here
_CS231N = "https://cs231n.stanford.edu/2017/syllabus.html"     # Spring 2017; each lecture row has a [video] link
_CS224N = "https://web.stanford.edu/class/cs224n/"
_CS236 = "https://deepgenerativemodels.github.io/"
_MIT = "https://introtodeeplearning.com/"
_FASTAI_P2 = "https://course.fast.ai/Lessons/part2.html"
_HF_LLM = "https://huggingface.co/learn/llm-course"
_HF_DIFF = "https://huggingface.co/learn/diffusion-course"
_HF_CV = "https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/"
_HF_AUDIO = "https://huggingface.co/learn/audio-course/"

VIDEOS = {
    "01": [
        ("Neural networks, Chapters 1–2: 'But what is a neural network?' and 'Gradient descent'", "3Blue1Brown",
         _3B1B + "gradient-descent", "The best visual intuition for neurons, layers and learning.", True),
        ("Introduction to Deep Learning (lecture 1)", "MIT 6.S191", _MIT, "A one-hour overview of the whole field.", False),
    ],
    "02": [
        ("What is backpropagation really doing? (Chapter 3)", "3Blue1Brown", _3B1B + "backpropagation",
         "Backprop without equations.", True),
        ("Backpropagation calculus (Chapter 4)", "3Blue1Brown", _3B1B + "backpropagation-calculus",
         "The chain-rule version, matching section 1 of the lesson.", True),
        ("makemore Part 4: Becoming a Backprop Ninja", "Andrej Karpathy", _KARPATHY,
         "Manual backprop through a real network, tensor by tensor.", False),
    ],
    "03": [
        ("The spelled-out intro to neural networks and backpropagation: building micrograd", "Andrej Karpathy",
         _KARPATHY, "Builds the same Value engine as this lesson, live, in 2.5 hours.", True),
    ],
    "04": [
        ("makemore Part 2: MLP", "Andrej Karpathy", _KARPATHY,
         "A full PyTorch training loop with mini-batches, splits and learning-rate tuning.", True),
    ],
    "05": [
        ("Lecture 7: Training Neural Networks II (optimizers, learning rates)", "Stanford CS231n (2017)", _CS231N,
         "SGD, momentum, RMSProp and Adam, with visualisations.", True),
        ("makemore Part 3: Activations & Gradients, BatchNorm", "Andrej Karpathy", _KARPATHY,
         "Initialisation and activation statistics in practice.", False),
    ],
    "06": [
        ("makemore Part 3: Activations & Gradients, BatchNorm", "Andrej Karpathy", _KARPATHY,
         "Diagnosing training with activation and gradient histograms; BatchNorm from scratch.", True),
        ("Lectures 6–7: Training Neural Networks I & II (BatchNorm, dropout, babysitting training)",
         "Stanford CS231n (2017)", _CS231N, "Regularization and debugging, lecture style.", False),
    ],
    "07": [
        ("Lecture 5: Convolutional Neural Networks", "Stanford CS231n (2017)", _CS231N,
         "Convolution, stride, padding and pooling, with diagrams.", True),
    ],
    "08": [
        ("Lecture 5: Convolutional Neural Networks", "Stanford CS231n (2017)", _CS231N, "CNN architecture basics.", False),
        ("Deep Computer Vision", "MIT 6.S191", _MIT, "A compact one-hour CNN lecture.", True),
    ],
    "09": [
        ("Lecture 9: CNN Architectures (AlexNet, VGG, GoogLeNet, ResNet)", "Stanford CS231n (2017)", _CS231N,
         "A tour of the architectures in this module.", True),
    ],
    "10": [
        ("Practical Deep Learning for Coders, Part 1 (lessons 1–2)", "fast.ai", "https://course.fast.ai/",
         "Transfer learning first, theory later: a great complement to this module.", True),
    ],
    "11": [
        ("Lecture 12: Visualizing and Understanding", "Stanford CS231n (2017)", _CS231N,
         "Feature visualisation, saliency maps and class activation maps.", True),
    ],
    "12": [
        ("Lecture 11: Detection and Segmentation", "Stanford CS231n (2017)", _CS231N,
         "R-CNN → Fast → Faster R-CNN, YOLO/SSD.", True),
    ],
    "13": [
        ("Lecture 11: Detection and Segmentation", "Stanford CS231n (2017)", _CS231N,
         "Semantic segmentation and upsampling (second half of the lecture).", True),
    ],
    "14": [
        ("Transformers, the tech behind LLMs (Chapter 5)", "3Blue1Brown", _3B1B + "gpt",
         "The big picture of a Transformer.", True),
        ("Attention in transformers, step-by-step (Chapter 6)", "3Blue1Brown", _3B1B + "attention",
         "Queries, keys and values, beautifully visualised.", True),
    ],
    "15": [
        ("Deep Generative Modeling", "MIT 6.S191", _MIT, "Autoencoders, VAEs and GANs in one lecture.", True),
        ("Lecture 13: Generative Models", "Stanford CS231n (2017)", _CS231N, "Autoencoders, VAEs and GANs.", False),
    ],
    "16": [
        ("CS236 lectures on latent variable models and VAEs (2023)", "Stanford CS236", _CS236,
         "The rigorous treatment of the ELBO.", True),
        ("Lecture 13: Generative Models (VAE part)", "Stanford CS231n (2017)", _CS231N, "A gentler introduction.", False),
    ],
    "17": [
        ("CS236 lectures on GANs (2023)", "Stanford CS236", _CS236, "GAN theory and training.", True),
        ("Lecture 13: Generative Models (GAN part)", "Stanford CS231n (2017)", _CS231N, "Intuition and DCGAN.", False),
    ],
    "18": [
        ("CS236 lectures on GANs and evaluating generative models (2023)", "Stanford CS236", _CS236,
         "Wasserstein GANs and how to evaluate generative models (FID).", True),
    ],
    "19": [
        ("From Deep Learning Foundations to Stable Diffusion (Part 2)", "fast.ai", _FASTAI_P2,
         "Builds diffusion models from scratch in code.", True),
        ("Diffusion Models Course (units 1–2)", "Hugging Face", _HF_DIFF, "Hands-on DDPM training with diffusers.", False),
        ("CS236 lectures on score-based and diffusion models (2023)", "Stanford CS236", _CS236, "The theory.", False),
    ],
    "20": [
        ("Lesson 9: Stable Diffusion", "fast.ai", "https://course.fast.ai/Lessons/lesson9.html",
         "How Stable Diffusion works, piece by piece.", True),
        ("Diffusion Models Course (guidance, Stable Diffusion units)", "Hugging Face", _HF_DIFF,
         "Guidance and Stable Diffusion pipelines in code.", False),
    ],
    "21": [
        ("Word vectors lectures (word2vec, GloVe)", "Stanford CS224n", _CS224N, "The standard NLP course's opening lectures.", True),
        ("makemore Part 2: MLP (character embeddings)", "Andrej Karpathy", _KARPATHY, "Embeddings learned inside a model.", False),
    ],
    "22": [
        ("Recurrent neural networks and LSTMs lectures", "Stanford CS224n", _CS224N, "RNN language models and gating.", True),
        ("Deep Sequence Modeling", "MIT 6.S191", _MIT, "RNNs to attention in one lecture.", False),
        ("Lecture 10: Recurrent Neural Networks", "Stanford CS231n (2017)", _CS231N, "Image captioning and char-RNNs.", False),
    ],
    "23": [
        ("Machine translation, seq2seq and attention lecture", "Stanford CS224n", _CS224N, "The encoder–decoder story.", True),
    ],
    "24": [
        ("Self-attention and Transformers lecture", "Stanford CS224n", _CS224N, "Transformers for language.", True),
        ("Attention in transformers, step-by-step (Chapter 6)", "3Blue1Brown", _3B1B + "attention", "Rewatch with masking in mind.", False),
    ],
    "25": [
        ("Let's build GPT: from scratch, in code, spelled out", "Andrej Karpathy", _KARPATHY,
         "The closest companion to this module.", True),
        ("Let's build the GPT Tokenizer", "Andrej Karpathy", _KARPATHY, "BPE from scratch.", True),
        ("Let's reproduce GPT-2 (124M)", "Andrej Karpathy", _KARPATHY, "Scaling up: efficient training details.", False),
        ("How might LLMs store facts (Chapter 7)", "3Blue1Brown", _3B1B + "mlp", "What the MLP blocks do.", False),
    ],
    "26": [
        ("LLM Course (chapters on Transformers, tokenizers and fine-tuning)", "Hugging Face", _HF_LLM,
         "The official guide to the library used in this module.", True),
        ("Pretraining lecture", "Stanford CS224n", _CS224N, "BERT-style and GPT-style pretraining.", False),
    ],
    "27": [
        ("Deep Dive into LLMs like ChatGPT", "Andrej Karpathy", "https://youtu.be/7xTGNNLPyMI",
         "Pretraining → SFT → RL, explained end to end (3.5 hours).", True),
        ("Prompting, instruction fine-tuning and RLHF lecture", "Stanford CS224n", _CS224N, "Alignment methods.", False),
        ("LLM Course (fine-tuning chapters)", "Hugging Face", _HF_LLM, "Hands-on fine-tuning recipes.", False),
    ],
    "28": [
        ("LLM Course (semantic search and embeddings sections)", "Hugging Face", _HF_LLM, "Embeddings and search in code.", True),
        ("Large Language Models explained briefly", "3Blue1Brown", _3B1B + "mini-llm", "A short refresher before building RAG.", False),
    ],
    "29": [
        ("Community Computer Vision Course: CLIP and relatives", "Hugging Face", _HF_CV + "clip-and-relatives/clip",
         "CLIP explained, with code.", True),
        ("Vision and Language lecture (slides)", "Stanford CS231n", "https://cs231n.stanford.edu/schedule.html",
         "Where contrastive pretraining fits among vision-language methods.", False),
    ],
    "30": [
        ("Recurrent Neural Networks lecture (image captioning and soft attention)", "Stanford CS231n", _CS231N,
         "Captioning with an encoder and a decoder, and attention over the image.", True),
        ("Community Computer Vision Course: multimodal tasks and models", "Hugging Face",
         _HF_CV + "tasks-models-part1", "Captioning, VQA and the models behind them.", False),
    ],
    "31": [
        ("Community Computer Vision Course: introduction to vision language models", "Hugging Face",
         _HF_CV + "vlm-intro", "How VLMs are put together.", True),
        ("Multimodality guest lecture", "Stanford CS224n", _CS224N, "Mixed-modal LLMs, from the NLP side.", False),
    ],
    "32": [
        ("Audio Course (chapter 1: working with audio data)", "Hugging Face", _HF_AUDIO + "chapter1/audio_data",
         "Waveforms, sampling and spectrograms.", True),
        ("Audio Course (chapter 5: automatic speech recognition)", "Hugging Face", _HF_AUDIO + "chapter5/introduction",
         "Speech recognition with Whisper in practice.", False),
    ],
}


def _title_md(title, url):
    return f"[{title}]({url})" if url else f"*{title}*"


def _ordered(items):
    return sorted(items, key=lambda p: not p[4])   # essential (⭐) first


def reading_cell_markdown(module_key: str) -> str | None:
    """Markdown for the 'Recommended reading & videos' cell appended to a lesson notebook."""
    papers, videos = PAPERS.get(module_key, []), VIDEOS.get(module_key, [])
    if not papers and not videos:
        return None
    lines = ["## 📚 Recommended reading and videos",
             "⭐ = start here. The others are optional, for going deeper."]
    if videos:
        lines += ["", "### 🎥 Videos"]
        for title, who, url, why, ess in _ordered(videos):
            lines.append(f"- {'⭐ ' if ess else ''}{_title_md(title, url)} ({who}). {why}")
    if papers:
        lines += ["", "### 📄 Papers",
                  "Tip: read the abstract, the figures and the conclusion first, then the method section.", ""]
        for title, who, url, why, ess in _ordered(papers):
            lines.append(f"- {'⭐ ' if ess else ''}{_title_md(title, url)} ({who}). {why}")
    return "\n".join(lines)


def readme_lines(module_key: str) -> list[str]:
    """Indented bullet lines inserted under a module in README.md."""
    lines = []
    for icon, label, items in (("🎥", "Recommended videos", VIDEOS.get(module_key, [])),
                               ("📄", "Recommended reading", PAPERS.get(module_key, []))):
        if items:
            lines.append(f"  - {icon} **{label}**")
            for title, who, url, _why, ess in _ordered(items):
                lines.append(f"    - {'⭐ ' if ess else ''}{_title_md(title, url)} ({who})")
    return lines
