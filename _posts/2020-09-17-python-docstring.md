---
comments: true
author: krishan
layout: post
categories: vscode
title: Host python code documentation using azure app service CI CD pipeline
description: How to upload python static html documentation created using mkdocs to an azure app service and enable CI CD pipeline using azure devops. 
---


This blog gives step by step guidance on.
1. Create  webapp for python flask
2. Deploy to azure.
3. Add AAD authentication.
4. Create CI CD pipeline using azure devops.
5. Create documentation using mkdocs.
6. upload mkdocs documentation to a separate static html webapp.
7. Set up CI CD for mkdocs 

clone the git repo with sample code. 

        git clone https://github.com/krishansubudhi/flaskWebApp.git
        cd flaskWebApp

# 1. Create a Virtual environment

        conda create -n webapp python==3.7
        conda activate webapp

# 2. create requirements.txt file. install dependencies.

        pip install -r requirements.txt
    
# 4. Instruction to create, test and deploy to azure app service
        
        https://docs.microsoft.com/en-us/azure/app-service/quickstart-python?tabs=powershell
# 5. Run locally
        
        Set-Item Env:FLASK_APP ".\application.py"
        flask run
# 6. Install azue CLI using powershell: [official docs](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows?view=azure-cli-latest&tabs=azure-powershell)

        Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi; Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'; rm .\AzureCLI.msi

This did not install CLI for me (probably permission issue). So run these commands one by one.

        Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi
        
        Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi'
        
        rm .\AzureCLI.msi

Restart powershell

        az login
# 7. Deploy flask web app to azure app service.

        az webapp up --sku F1 -n krishan-test-flask

This failed for me. Probably because it tries to create a new resource group in a default subscription which requires owner access. 
        
I set default subscription and passed a resource group where I have owner access. 

        az account set --subscription "<subscription id>"
        az webapp up -n krishan-test-flask --resource-group <existing-rg-name>

It takes some time but finally created the webapp. I could view my web app in the azure portal's app service section

        {
        "URL": "http://krishan-test-flask.azurewebsites.net",
        "appserviceplan": "krkusuk_asp_Linux_centralus_0",
        "location": "centralus",
        "name": "krishan-test-flask",
        "os": "Linux",
        "runtime_version": "python|3.7",
        "runtime_version_detected": "-",
        "sku": "PREMIUMV2",
        "src_path": "C:\\Users\\krkusuk\\repos\\flaskWebApp"
        }
The webapp is up and running

# 8. Add AAD authentication to webapp
This will restrict access to people with from your company.. 
https://docs.microsoft.com/en-us/azure/app-service/configure-authentication-provider-aad#-configure-with-express-settings

Did not understand point 5.

# 9. Make changes and redeploy


First make some change to the hello() function

```python
        return "Hello World Again!"
```

        az webapp up


https://krishan-test-flask.azurewebsites.net/

# 10. Automatic deployment
We want the app service to be updated whenever we checkin new changes.

The CLI command above creates a zip of the webapp code and uploads the content to some location. This is abstracted. We want this CLI step to be executed through a CI CD pipeline.

CI is for building the deployment package.
CD is for deployment to app sevice or other such services.

Very good documentation
https://docs.microsoft.com/en-us/azure/devops/pipelines/get-started/pipelines-sign-up?view=azure-devops

Build CICD pipeline following above documentation. CI and CD will be two different stages.

If getting authentication error on subscription , go to project settings-> service connection -> Azure Resource Manager -> create new service connection

# 11. Make project docs using md files and docstring
https://www.mkdocs.org/
https://pawamoy.github.io/mkdocstrings/handlers/python/

        cd docs
        mkdocs serve

        mkdocs build
The htmls arepresent in docs/site now.

Looks like flask was not required afterall.
        cd docs/site
        az webapp up -n krishan-test-html --html --resource-group hackathon #resource group bert-base threw error saying it's not configured for windows

https://krishan-test-html.azurewebsites.net/

It shows the documents now.

It can be handy to include the flask section in future though when we want to add some functionality.


# 12. Use azure pipeline for mkdocs project
Since this is a new webapp, let's creat another azure pipeline.

Go to your devops account-> pipeline->new pipeline->Github-> select application -> python to linux webapp on azure - >

If getting authentication error with the workspace, go to service connection in project settings, delete existing connection and create a new one with proper resource group

select pythonpackage in configure. We will manually modify the yaml and change the deployment to azure app service. For this demo, copy azure-pipelines.yml content into the pipeline.

Run pipeline. Solve any errors in the pipeline.

Make changes in the index.md and mkdocs.yml and checkin. See changes automatically being deployed by CI pipeline.

Open https://krishan-test-html.azurewebsites.net/ and see latest changes reflected after the pipeline runs automatically.
