---
comments: true
author: krishan
layout: post
categories: python, jax, tpu
title: Using TPU runtimes in colab for JAXs
description: Explains how functools partial works with simple examples. 
---

In colab, TPU runtime can be selected in the menu option
`Runtime -> change runtime type -> Harware Acclerator`

By default Jax does not recognize TPUs. We need to set up the cloud TPU for that.

Here is the default behaviour in a tpu runtime.


```python
import jax
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
jax.devices()
```

    cpu

    [CpuDevice(id=0)]



To use TPUs we need to follow the steps,

1. Restart runtime `Runtime -> Restart Runtime`
2. Set up cloud tpu before importing jax. 

As per the book *Jax in Action*
> It changes the JAX XLA backend to TPU and links the Jax backend to TPU host.

As per Bard, (*prompt = jax.tools.colab_tpu.setup_tpu() - what does this do?*

> The function jax.tools.colab_tpu.setup_tpu() sets up a TPU device for use in Colab. It does this by connecting to the TPU service and initializing the TPU driver. Once the TPU driver is initialized, it can be used to run JAX code on the TPU.

> To use jax.tools.colab_tpu.setup_tpu(), you first need to make sure that the Colab Runtime is set to Accelerator: TPU. Then, you can run the following code to set up the TPU:


```python
# set up cloud TPU
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()

import jax
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
jax.devices()
```

    tpu

    [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0),
     TpuDevice(id=1, process_index=0, coords=(0,0,0), core_on_chip=1),
     TpuDevice(id=2, process_index=0, coords=(1,0,0), core_on_chip=0),
     TpuDevice(id=3, process_index=0, coords=(1,0,0), core_on_chip=1),
     TpuDevice(id=4, process_index=0, coords=(0,1,0), core_on_chip=0),
     TpuDevice(id=5, process_index=0, coords=(0,1,0), core_on_chip=1),
     TpuDevice(id=6, process_index=0, coords=(1,1,0), core_on_chip=0),
     TpuDevice(id=7, process_index=0, coords=(1,1,0), core_on_chip=1)]


