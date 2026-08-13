python -m pip --no-cache-dir install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu128

make WORKERS=16
