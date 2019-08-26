---
author: krishan
layout: post
categories: deeplearning
title: Azure Machine Learning Tutorial
---

[Original Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-train-models-with-aml)
Video link: https://channel9.msdn.com/Events/Connect/Microsoft-Connect--2018/D240/

Start by creating a free subscription in azure portal : https://portal.azure.com

#### Create workspace


```python
from azureml.core import Workspace,Experiment, Run
subscription_id='####'
workspace_name= 'krishan_test_ws'
resource_group='krishan'

ws = Workspace.create(subscription_id, resource_group, workspace_name)
```

#### Save Workspace config


```python
ws.write_config()

!dir
```

## Train image classification model using AML
https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-train-models-with-aml

#### Load workspace
Either of the below methods will work


```python
#ws = Workspace(subscription_id, resource_group, workspace_name)
#ws = Workspace.list(subscription_id)[workspace_name]
ws = Workspace.from_config()

print(ws.name, ws.location, ws.resource_group, sep = '\t')
```

####  Create an experiment


```python
experiment_name = 'sklearn-mnist'

exp = Experiment(ws, experiment_name)
exp
```

#### Create compute 


```python
from azureml.core.compute import AmlCompute,ComputeTarget

# choose a name for your cluster
compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpucluster")
compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

# This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")


if compute_name in ws.compute_targets:
    compute_targets = ws.compute_targets[compute_name]
    if compute_targets and type(compute_targets) is AmlCompute:
        print('Found compute tarket in ws already. Use it.', compute_name)
else:
    print('creating new compute target ...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size =vm_size,
                                                               min_nodes = compute_min_nodes,
                                                               max_nodes = compute_max_nodes)

    #Create cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config )
    # can poll for a minimum number of nodes and for a specific timeout.
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes = 20)

    #For more detailed view of current AmlCompute status, use get_status()
    print(compute_target.get_status().serialize())
```

### Explore data
1. Download the MNIST dataset.
1. Display some sample images.
1. Upload data to the cloud.

#### Download data


```python
import urllib.request
import os

data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok = True)

urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename=os.path.join(data_folder, 'train-images.gz'))
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename=os.path.join(data_folder, 'train-labels.gz'))
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename=os.path.join(data_folder, 'test-images.gz'))
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename=os.path.join(data_folder, 'test-labels.gz'))
```

#### Display images 


```python
# make sure utils.py is in the same directory as this code
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data

# note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the model converge faster.
X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0
y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)

# now let's show some randomly chosen images from the traininng set.
count = 0
sample_size = 30
plt.figure(figsize = (16, 6))
for i in np.random.permutation(X_train.shape[0])[:sample_size]:
    count = count + 1
    plt.subplot(1, sample_size, count)
    plt.axhline('')
    plt.axvline('')
    plt.text(x=10, y=-10, s=y_train[i], fontsize=18)
    plt.imshow(X_train[i].reshape(28, 28), cmap=plt.cm.Greys)
plt.show()
```

#### Upload data to the cloud
Now make the data accessible remotely by uploading that data from your local machine into Azure. Then it can be accessed for remote training. The datastore is a convenient construct associated with your workspace for you to upload or download data. You can also interact with it from your remote compute targets. It's backed by an Azure Blob storage account.

The MNIST files are uploaded into a directory named mnist at the root of the datastore:


```python
ds = ws.get_default_datastore()
print(type(ds).__name__, ds.datastore_type, ds.account_name, ds.container_name)
```


```python
ds.upload(src_dir=data_folder, target_path='mnist', overwrite=True, show_progress=True)
```

### Train on a remote cluster
For this task, submit the job to the remote training cluster you set up earlier. To submit a job you:

1. Create a directory
1. Create a training script
1. Create an estimator object
1. Submit the job


```python
import os
script_folder  = os.path.join(os.getcwd(), "sklearn-mnist")
os.makedirs(script_folder, exist_ok=True)
```


```python
%%writefile $script_folder/train.py

import argparse, os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from azureml.core import Run
from utils import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--regularization', type=str, dest='reg', help='regularization rate')
args = parser.parse_args()

data_folder = args.data_folder
print ('data folder:', data_folder)

'''
from argparse import Namespace
args = Namespace(
    data_folder = 'mnist',
    reg = 0.01
)
args.reg
'''

X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0
y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)

X_train.shape, X_test.shape,

print('train a logistic regression model at reg = ',args.reg)
clf = LogisticRegression(C = 1.0/np.float(args.reg), random_state=42, multi_class='auto', solver='liblinear', max_iter=3)
clf.fit(X_train, y_train)

print('Predict the test set')
y_hat = clf.predict(X_test)

#accuracy
acc = np.average(y_hat == y_test)
print('Accuracy is', acc)

# get hold of the current run
run = Run.get_context()

run.log('regularization_rate', np.float(args.reg))
run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok = True)

#file saved in outputs folder is automatically uploaded to experiment record
joblib.dump(value=clf, filename='outputs/sklearn_mnist_model.pkl')

```


```python
import shutil
shutil.copy('utils.py', script_folder)
```

### Create Estimator


```python
from azureml.train.sklearn import SKLearn

script_params = {
    '--data-folder':ds.path('mnist').as_mount(),
    '--regularization':0.5
}

est = SKLearn(source_directory = script_folder,
             script_params = script_params,
             compute_target = compute_targets,
             entry_script = 'train.py')
```

#### Submit the job to the cluster
Run the experiment by submitting the estimator object:


```python
run = exp.submit(config = est)
run
```


```python
from azureml.widgets import RunDetails
RunDetails(run).show()
```


```python
run.wait_for_completion(show_output=False) # specify True for a verbose log
```


```python
print(run.get_metrics())
```


```python
print(run.get_file_names())
```

####  register the model


```python
# register model
model = run.register_model(model_name='sklearn_mnist', model_path='outputs/sklearn_mnist_model.pkl')
print(model.name, model.id, model.version, sep = '\t')
```


```python
# optionally, delete the Azure Machine Learning Compute cluster
compute_targets.delete()
```
