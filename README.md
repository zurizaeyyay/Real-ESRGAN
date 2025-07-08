# Real-ESRGAN
PyTorch implementation of a Real-ESRGAN model trained on custom dataset. Forked from [Repo](https://github.com/ai-forever/Real-ESRGAN)

> Which partially uses code from the original researcher [original repository](https://github.com/xinntao/Real-ESRGAN)

Real-ESRGAN is an upgraded [ESRGAN](https://arxiv.org/abs/1809.00219) trained with pure synthetic data is capable of enhancing details while removing annoying artifacts for common real-world images. 

- [Paper (Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data)](https://arxiv.org/abs/2107.10833)
- [Original implementation](https://github.com/xinntao/Real-ESRGAN)

### Installation

```bash
pip install git+https://github.com/zurizaeyyay/Real-ESRGAN.git
```

### Usage

---

Basic usage:

```python
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_to_image = 'inputs/lr_image.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('results/sr_image.png')
```

### Examples

---

Low quality image:

![](examples/inputs/lr_image.png)

Real-ESRGAN result:

![](examples/results/sr_image.png)

---

Low quality image:

![](examples/inputs/lr_face.png)

Real-ESRGAN result:

![](examples/results/sr_face.png)

---

Low quality image:

![](examples/inputs/lr_lion.png)

Real-ESRGAN result:

![](examples/results/sr_lion.png)
