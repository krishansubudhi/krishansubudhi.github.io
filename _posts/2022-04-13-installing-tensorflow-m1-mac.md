---
comments: true
author: krishan
layout: post
categories: tensorflow
title: Installing tensorflow on M1 Macs
description: Step by step instructions for installing tensorflow on M1 Macbooks with Apple Silicon.
---
This blog provides step by step instructions for installing tensorflow on M1 Macbooks with Apple Silicon.

`pip install tensorflow` does not work on M1 Macs.

```
ERROR: Could not find a version that satisfies the requirement tensorflow (from versions: none)
ERROR: No matching distribution found for tensorflow
```
There is an official Apple instruction page that I have not tried. 

[Getting Started with tensorflow-metal PluggableDevice](https://developer.apple.com/metal/tensorflow-plugin/)

Tensorflow is available as a different pypi package named `tensorflow-macos` for M1 Macs. But doing `pip install tensorflow-macos` resulted in the following error.
```
ERROR: Could not build wheels for h5py which use PEP 517 and cannot be installed directly.

<--Similar error for another wheel GRPCIO -->
```

Final Solution that worked for me.
```
conda create -n tf python==3.8
conda activate tf

pip install --upgrade setuptools
conda install HDF5

# bottom 3 lines for GRPCIO dep failure. https://stackoverflow.com/questions/66640705/how-can-i-install-grpcio-on-an-apple-m1-silicon-laptop

export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
pip install firebase-admin

pip install tensorflow-macos

Successfully installed absl-py-1.0.0 astunparse-1.6.3 google-auth-oauthlib-0.4.6 google-pasta-0.2.0 markdown-3.3.6 requests-oauthlib-1.3.1 tensorboard-2.8.0 tensorflow-macos-2.8.0
```