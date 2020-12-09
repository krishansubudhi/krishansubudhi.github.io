---
comments: true
author: krishan
layout: post
categories: vscode
title: Runing ML Training code on a VM
description: How to move a local machine learning traning code to a virtual machine and develop there.
---


Krishan Subudhi 12/08/2020

---

In this blog we will ssh into a GPU linux machine and run our ML code using jupyter notebook on the VM.

This is mostly required when we want to run our ML traning code in a GPU machine to quickly do few train iteration before moving to more powerful orchestrators like Azure Machine Learning or Amazon Sage Maker.

These are the steps I followed. This avoids the need to boot a DSVM and does not depend on visual studio code. What we need is only powershell, browser and source code in local machine.  
## add ssh key to remote machine
This will remove the need to enter password for the VM at every step
From local machine powershell

    > ssh-keygen

    > ssh elr@104.210.193.245 echo $key>>~/.ssh/authorized_keys
    elr@104.210.193.245's password:

    > ssh elr@104.210.193.245 cat ~/.ssh/authorized_keys
    Enter passphrase for key 'C:\Users\krkusuk/.ssh/id_rsa':

Change passphrase of your ssh key later
    > ssh-keygen -p

## Transferring code
### Option 1: copy folder containing code to remote machine
    scp -r C:\Users\krkusuk\repos\ELR\sources\dev\SubstrateInferences\ELR2-Scenarios\classification elr@104.210.193.245:/home/elr

Optinoal: Install any dependencies by sshing into the VM
    
    > ssh elr@104.210.193.245
    
    $ cd /home/elr/classification
    $ pip install -r requirements.txt

This need files to by synced again with remote machine. But provides a quick and easy to to test code in VM. 

Sync back with local files and checkin

    > scp -r C:\Users\krkusuk\repos\ELR\sources\dev\SubstrateInferences\ELR2-Scenarios\ elr@104.210.193.245:/home/elr/classification
    > cd C:\Users\krkusuk\repos\ELR\sources\dev\SubstrateInferences\ELR2-Scenarios\
    > git add -u
    > git commit -m "gpu changes"
    > git push

### Option 2: git clone in remote machine
Get access token and do git clone in remote machine. 
Do this if you want frequent check ins for your changes and do nt want to lose progress.

## Open jupyter notebook and tunnel using ssh
This will remove the need to open port 8000 in remote machine


1. Run this on your `remote-machine`
    jupyter notebook --no-browser --port=8898
2. Run this on your `local-machine`

    This probably creates a tunnel to remote machine through ssh

        > ssh -N -f -L 127.0.0.1:8898:127.0.0.1:8898  elr@104.210.193.245
3. Type this in the browser on your `local-machine`
    http://127.0.0.1:8898/

## Open visual studio code to make changes in remote machine
Open vs code and install `remote ssh` extension. Connect to the remote host and open the corresponding code in local vs code. Any changes done in vs code will be reflected in remote machine.
 
## References:

https://kawahara.ca/how-to-run-an-ipythonjupyter-notebook-on-a-remote-machine/
https://stackabuse.com/copying-a-directory-with-scp/