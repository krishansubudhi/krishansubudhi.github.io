---
comments: true
author: krishan
layout: post
categories: deeplearning
title: Using Tensorboard effeciently in AzureML
description: Using tensorboard client effeciently in azureml
---

Begin logging stats to tensorboard from your training scripts by following this [AzureML documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-monitor-tensorboard).

This tutorial focuses on improving the client side experiment. Azure SDKs give basic functionality to view tensorboard logs in local machine.

This tutorial wil enable
1. Viewing tensorboard logs across experiments and workspaces.
2. Running tensorboard in a VM and share links
3. More control over the tesorboard process

The approach is to download the experiment logs using azureml-sdk APIs and starting tensorboard manually.

First, [install azureml sdk](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#local)

    conda install notebook ipykernel
    ipython kernel install --user --name aml --display-name "Python (aml)"
    pip install azureml-sdk[notebooks]

In case of pyyaml error

    pip install --upgrade azureml-sdk\[notebooks\] --ignore-installed PyYAML

Check current running processes in port 6006. Import azureml packages.

```python
! ps -axu | grep 6006

from azureml.core import Experiment, Workspace
from azureml.core.workspace import Workspace
from azureml.tensorboard import Tensorboard

```


```python
expr_runs = []
```

# Workspace 1


```python
subscription_id = None # enter from portal
resource_group = None # enter from portal
workspace_name = None # enter from portal
ws = Workspace(subscription_id, resource_group, workspace_name)
print(ws)
expr_runs.extend([
    (ws, 'WS1_myamlexp1','latest'),
    (ws, 'WS1_myamlexp2','running')
])
```

# Workspace 2


```python
subscription_id = None # enter from portal
resource_group = None # enter from portal
workspace_name = None # enter from portal
ws = Workspace(subscription_id, resource_group, workspace_name)
print(ws)
expr_runs.extend([
    (ws, 'WS2_myamlexp1','running'),
    (ws, 'WS2_myamlexp1','WS2_myamlexp_12324')
    
])
```

# Fetch run ids from azure
```python
runs = []
print([e[1] for e in expr_runs])
max_runs = 10
for ws, experiment_name, run_id in expr_runs:
    exp =Experiment(workspace=ws, name=experiment_name, _create_in_cloud=False)
    for run in exp.get_runs():
        if run_id == 'running' and run.status == 'Running':
            runs.append(run)
        elif run_id == 'latest':
            runs.append(run)
            break;
        elif run_id == 'all' or run.id == run_id: 
            runs.append(run)
            if len(runs) >= max_runs:
                break
        elif run.id == run_id:
            runs.append(run)
            break

print(f'Found {len(runs)} runs')
```

# Set Port


```python
port = '6006'
local_root = 'logs_'+port
```

# Delete previous runs


```python
! ls $local_root
! rm -rf $local_root
! mkdir $local_root
```

# Download logs to local disc

#### Azure API documentations

[get_file_names](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py#get-file-names--)

[download_file for a single file](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py#download-file-name--output-file-path-none-)
  
[download_files for bulk download](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py#download-files-prefix-none--output-directory-none--output-paths-none--batch-size-100--append-prefix-true-)


```python

def download():
    import os
    for run in runs:
        output = os.path.join(local_root, run.id)
        print(f'\nChecking event files for {run.id} status = {run.status}, output = {output}')
        run.download_files(prefix = 'logs/',output_directory=output)
    print(os.listdir(local_root))

download()
```

# Run tensorboard

To keep it running in background, run this in a shell and don't run the below commands. But unless restarted, tensorboard keeps showing of deleted logs too.


    (azureml_py36_tensorflow) elr@elr-tensorboard:~/notebooks$ nohup tensorboard --logdir krishan/logs_6006 --host 0.0.0.0 --port 6006 & 2> krishan/tb_run.log


```python
!dir $local_root
!tensorboard --logdir $local_root --host 0.0.0.0 --port $port
```

Open `https://<IP>:6006`  in your browser to view tensorboard graphs

Stop this cell and run two last cells to refresh the graphs and download new data.

If not intending to stop tensorboard while refreshing, run tensorboard in background for your specific port.
