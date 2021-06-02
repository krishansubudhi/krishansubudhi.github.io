---
comments: true
author: krishan
layout: post
categories: azure
title: File transfer from blob storage using azure cli
description: Step by step instructions on how to move files and directories to and from azure blob storage.
---


Downloading and uploading files from blob should be simple but I often come across a lot of errors while doing so. This blob documents the steps that have worked for me. 

This is how azure blob fits into the hierarchy

* microsoft account (your microsoft account like example@hotmail.com)
    * subscription (each azure account comes with subscription id. Generally one subscription for each team)
        * storage account (storage account contains multiple containers)
            * blob storage container (this is where the actual files are stored)

## Azure CLI
Install azure cli using the instructions mentioned [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-linux?pivots=apt).

this is the sdk/tool through which we can download/upload/delete or list files and folders in blob storage.

Do these for convenience 

    az login
    az account set --subscription <subscription id>
    export AZURE_STORAGE_ACCOUNT=krishan
    export AZURE_STORAGE_KEY=<storage account key from azure portal>

## 1. List blob

    az storage blob  list -c <container-name>

## Upload
    az storage blob upload  -f file.txt -c <container-name> -n file.txt

## Download
    az storage blob download  -f file.txt -c <container-name> -n file.txt

## upload dir

Let's say we want to save model checkpoints

    az storage blob upload-batch -d mtl/checkpoints -s /home/azureuser/MTL/checkpoints

## Viewing the file structure

Download and install [Azure Storage Explorer](https://docs.microsoft.com/en-us/azure/vs-azure-tools-storage-manage-with-storage-explorer?tabs=windows) if you prefer GUI.

## Reference:

[Azure Documentation](https://docs.microsoft.com/en-us/cli/azure/storage/blob?view=azure-cli-latest#az_storage_blob_list)
