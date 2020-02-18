---
comments: true
author: krishan
layout: post
categories: python
title: Resize image using Python
---


Resize image in python using pillow library

    pip install pillow


Code

```python
import PIL
from PIL import Image
MAXWIDTH = 450

filename = '1.jpg'

img = Image.open(filename)

image_bytes = len(img.fp.read())
print(f'image size = {image_bytes}')
s= img.size
ratio = MAXWIDTH/s[0];

print(f'ratio = {ratio}')

newimg = img.resize((int(s[0]*ratio), int(s[1]*ratio)) , Image.NEAREST)
newimg.save('resized_1.jpg')
print(f'new image size = {newimg.size}')
```
