COMP9517 Group Project - Deep Learning Methods (Wheat Crop Segmentation)

This folder contains three deep learning notebooks for wheat crop segmentation on the EWS dataset: U-Net, DeepLabV3+, and SegFormer.

---

Dataset Setup

Download the EWS dataset from:
https://www.research-collection.ethz.ch/entities/researchdata/165d22fc-6b0f-4fc3-a441-20d8bdc50a70

Extract and place the train/, val/, and test/ folders in the same directory as the notebooks. Each folder contains paired .png image and _mask.png files.

---

Requirements

- Python 3.9+
- CUDA-compatible NVIDIA GPU (required)
- PyTorch >= 2.6.0

Install dependencies

Step 1 - Install PyTorch with CUDA support (check your CUDA version via nvidia-smi):

    # Example for CUDA 12.1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # Example for CUDA 12.4
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Step 2 - Install other dependencies:

    pip install segmentation-models-pytorch albumentations opencv-python matplotlib tqdm transformers

Verify installation:

    import torch, transformers
    print(torch.__version__)         # should be >= 2.6.0
    print(transformers.__version__)
    print(torch.cuda.is_available()) # should be True

If PyTorch version is below 2.6, upgrade with:
    python -m pip install --upgrade torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    (replace cu124 with your own CUDA version)

---

How to Run

Open each notebook in JupyterLab or VS Code and run all cells from top to bottom.
Each notebook is self-contained and will:

    1. Load and preprocess the EWS dataset
    2. Train the model on the training set
    3. Select the best checkpoint based on validation IoU
    4. Evaluate on the test set and print Precision, Recall, F1, and IoU
    5. Display qualitative result visualisations

Training takes approximately 3-6 minutes per notebook on a modern GPU.

---

Third-Party Libraries and Sources

- segmentation-models-pytorch : U-Net and DeepLabV3+ model implementations
  https://github.com/qubvel/segmentation_models.pytorch

- transformers (Hugging Face) : SegFormer model implementation
  https://huggingface.co/docs/transformers

- albumentations : Data augmentation pipeline
  https://albumentations.ai

All models use pretrained ImageNet weights loaded automatically by the respective libraries.
No pretrained weight files are included in this submission.

---

Reproducibility Note

A fixed random seed (SEED = 42) is set at the start of each notebook. However, completely
identical results across different hardware, PyTorch versions, or platforms are not guaranteed.
See: https://docs.pytorch.org/docs/stable/notes/randomness.html
