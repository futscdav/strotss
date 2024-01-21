# STROTSS

See the original code and links to paper at https://github.com/nkolkin13/STROTSS

Usage:
```
python strotss.py <content> <style> [--weight 1.0] [--output strotss.png] [--device "cuda:0"]
```

<p align="center">
  <img src="content.jpg" width="350" title="Content">
  <img src="style.png" width="350" alt="Style">
  <img src="strotss.png" width="350" alt="Result">
</p>

## Masked guidance
This version is an implementation of guidance-free style transfer. The original paper shows a path towards mask-based guidance.
In that scenario, the user needs to provide correspondences for style transfer source and destinations.
Hassan Rasheed extended this implementation to also support that case, so if you are looking for such functionality, you can try his implementation at https://github.com/hassanrasheedk/fast-strotss.
