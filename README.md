Run Instructions
1️⃣ Test the model forward pass
baash: python test_msffa.py

2️⃣ Run inference on a sample foggy image (untrained MSFFA)
bash: python run_on_image.py

Output image will appear as a dark/noisy enhanced version — this is expected until training.

#Requirements

Install dependencies:
bash: pip install torch torchvision pillow
