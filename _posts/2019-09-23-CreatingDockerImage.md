---
author: krishan
layout: post
comments: true
categories: development
title: How to create a new docker image
---

# Steps to create, test and push a docker image

1. Download and install [Docker Desktop](https://docker.com/get-started). If on Linux, download Docker Engine - Community.
2. Create a Docker file or download one. [Example](https://github.com/microsoft/AzureML-BERT/blob/master/pretrain/PyTorch/docker/Dockerfile#L12)
    
        >notepad dockerfile

2. Login to docker hub

        >docker login

3. Make changes to docker file and build it

        >docker build --rm -t krishan_apex:latest .
        
        Sending build context to Docker daemon  3.584kB
        **** installs a bunch of packages ***
        **** installs some more packages ***
        Successfully built 4a4cc017fe68

4. See the list of docker images

        > docker images
        REPOSITORY                           TAG                                        IMAGE ID            CREATED             SIZE
        krishan_apex                         latest                                     4a4cc017fe68        2 minutes ago       6.33GB


5. Create a repository in [docker hub](https://hub.docker.com/)

6. Tag your image with <username>/<reponame>:<tag>
    
        > docker tag krishan_apex krishansubudhi/pytorch_apex:latest

7. Run the container on your machine using the image you just created.

        > docker run -it -d -p 5000:5000  krishansubudhi/pytorch_apex:latest

8. Check all running docker containers

        > docker ps
        
        CONTAINER ID        IMAGE                                COMMAND             CREATED             STATUS              PORTS                                                  NAMES
        55c7cfc80e35        krishansubudhi/pytorch_apex:latest   "/bin/bash"         15 seconds ago      Up 13 seconds       5001/tcp, 8883/tcp, 0.0.0.0:5000->5000/tcp, 8888/tcp   angry_elbakyan


9. Log in to terminal of runing container and run simple linux commands. 
        
        docker attach 55c7cfc80e35

10. Push the image to dockerhub repo to be accessed by everyone. This takes a long time.
        
        docker push krishansubudhi/pytorch_apex:latest

### Reference : 

[IBM tutorial](https://developer.ibm.com/tutorials/accessing-dockerhub-repos-in-iks/)
[Official documentation](https://docs.docker.com/docker-hub/)
