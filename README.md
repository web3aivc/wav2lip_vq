Wav2lip in a compact Vector Quantized (VQ) space

- VQGAN 
- https://github.com/CompVis/taming-transformers
    - debugging custom models #107
    - fine-tune based on [vqgan_imagenet_f16_1024]
    - https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/
- image_size = 256
- syncnet_vq.py
  - face_encoder: (B, T x 256, 16, 16) -> (B, 512, 1, 1)
  - audio_encoder: (B, 1, 80, 16) -> (B, 512, 1, 1)
- color_syncnet_train_vq.py
  - vqgan config / ckpt