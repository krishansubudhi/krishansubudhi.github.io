---
comments: true
author: krishan
layout: post
categories: datascience
title: Challenges of using HDInsight for pyspark
description: Challenges and approaces to submit job in VS code with links, results and opinions
---

The goal was to do analysis on the following dataset using Spark without download large files to local machine.

https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3

The file size is around 2 GB. I had been running all my analysis in local spark cluster before. I started to search for alternatives. HDInsight is azure's solution to run distributed big data analysis jobs. HDInsight also has spark support.

## HDI spark job submission ways

1. Local machine. Jupyter notebook or spark submit

    File is too large. Will be too slow and require downloading large file to local machine.

2. [Cloud - Azure HDinsight](https://docs.microsoft.com/en-us/azure/hdinsight/spark/apache-spark-jupyter-spark-sql-use-portal)
    
    Number of ways to submit spark job - [Azure documentation](https://docs.microsoft.com/en-us/archive/blogs/azuredatalake/spark-job-submission-on-hdinsight-101)
    
    1. [Interactive jupyter shell in HDI cluster](https://docs.microsoft.com/en-us/azure/hdinsight/spark/apache-spark-load-data-run-query)

        Quickest way, but I was not able to create Jupyter notbooks in HDI cluster

    ![jupyter error hdinsight](/assets/hdi/jupyter-error.jpg)

    2. [Run jupyter locally and connect to HDI cluster](https://docs.microsoft.com/en-us/azure/hdinsight/spark/apache-spark-jupyter-notebook-install-locally)

        Lots of steps - but can be one time effort if everything works and you like jupyter a lot.

    3. [Write code and submit batch job through API](https://docs.microsoft.com/en-us/azure/hdinsight/spark/apache-spark-livy-rest-interface)

        Requires JAR creation. No instruction for python
    3. [Use hdi cluster interactive pyspark shell](https://docs.microsoft.com/en-us/azure/hdinsight/spark/apache-spark-shell)

        Pros: No installations required. 
        
        Cons: Code needs to be transferred from local machine to machine with pyspark shell. Easiest way to speed up the copy  will be by connecting local vscode with this machine.
        
        This is ok for quick testing. But not for day to day work. Still if nothing works, this will be my last resort. 

    4. [Visual studio code for pyspark](https://azure.microsoft.com/en-us/blog/run-your-pyspark-interactive-query-and-batch-job-in-visual-studio-code/)

        Probably the second most easiest way after jupyter notebook. A lot is abstracted out and you are dependent a lot on visual studio code.
        
        Also the code does not need to be transferred from local machine to spark cluster manually as vscode will take care of that.

## HDI Submission : VS Code
So I finally decided to use visual studio to submit my spark job.

1. I created a spark cluster in my azure subscription.

    When a spark cluster is created, a storage account and container is also required. That container contains lots of sample code and data.

    ![hdi container](/assets/hdi/container.jpg)

2. I followed [this](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-for-vscode) documentation from azure to install VSCode extensions for pyspark job submission in azure HDI. This also supports interactive query. 


    1. Installed VSCode extension

        ![vscode extension](/assets/hdi/vscode-extension.jpg)

    2. connected the cluster as per documentation. But while submitting my spark job, I got this error

            [2020-1-12:10:47:59] [Info] log is: 	 user: livy,20/01/12 18:47:52 INFO ShutdownHookManager: Shutdown hook called,20/01/12 18:47:52 INFO ShutdownHookManager: Deleting directory /tmp/spark-1f2ef8d2-b808-4c92-b361-9efa0e4b62d1,20/01/12 18:47:52 

    Now this was getting frustrating. I am going to try the method 4 (Use hdi cluster interactive pyspark shell).

## HDI submission : pyspark shell

Followed [this](https://docs.microsoft.com/en-us/azure/hdinsight/spark/apache-spark-shell) documentation.

1. Opened powershell. 
        
        ssh  sshuser@krishan-spark.azurehdinsight.net

        Error:

        ssh: connect to host krishan-spark.azurehdinsight.net port 22: Connection timed out

    Went to the spark cluster in the portal and corrected the host name.

        ssh sshuser@krishan-spark-ssh.azurehdinsight.net 

        sshuser@hn0-krisha:~$ pyspark

        Welcome to
            ____              __
            / __/__  ___ _____/ /__
            _\ \/ _ \/ _ `/ __/  '_/
        /__ / .__/\_,_/_/ /_/\_\   version 2.4.0.3.1.2.2-1
            /_/

        Using Python version 2.7.12 (default, Jul  2 2016 17:42:40)
        SparkSession available as 'spark'.
        >>>

    Quick testing with custom data
        >>> data = [['krishan', 'microsoft']]
        >>> df = spark.createDataFrame(data).show()

        +-------+---------+
        |     _1|       _2|
        +-------+---------+
        |krishan|microsoft|
        +-------+---------+

    Testing with existing files in azure storage.

        >>> df = spark.read.parquet('/example/data/people.parquet')
        >>> df.show()
        +---+-----+
        |age| name|
        +---+-----+
        | 22|Ricky|
        | 36| Jeff|
        | 62|Geddy|
        +---+-----+

Finally something is working!! Now I just need to upload my file to azure storage liked to my spark cluster and run commands in shell.


I opened an azure cloud shell. Logged in to my azure account. 
    
    azcopy cp "https://data.sfgov.org/api/views/nuek-vuh3/rows.csv?accessType=DOWNLOAD" "https://krishansparkhdistorage.blob.core.windows.net/spark-container1"

    Error:
    failed to perform copy command due to error: cannot start job due to error: cannot scan the path /home/krishan/https:/data.sfgov.org/api/views/nuek-vuh3/rows.csv?accessType=DOWNLOAD, please verify that it is a valid.

I searched in Bing for ways to transfer data directly from a url to azure storage using the search query *azcopy to download file directly from url* . But I only found ways to transfer data from s3 to azure. I wonder why such a simple feature is not present in azure.

I wanted to avoid download data to my local machine.

I found a workaround and used the cloud shell to download data to cloud shell storage first and then to azure storage.

    azcopy login

    azcopy cp "https://data.sfgov.org/api/views/nuek-vuh3/rows.csv?accessType=DOWNLOAD" "https://krishansparkhdistorage.blob.core.windows.net/spark-container1" -O sffiredata.csv

    krishan@Azure:~$ ls -l
    total 1856644
    lrwxrwxrwx 1 krishan krishan         22 Jan 12 19:15 clouddrive -> /usr/csuser/clouddrive
    -rw-r--r-- 1 krishan krishan 1899340731 Jan 12 11:06 sffiredata.csv

    
    krishan@Azure:~$ azcopy copy sffiredata.csv https://krishansparkhdistorage.blob.core.windows.net/spark-container1/
    
    RESPONSE Status: 403 This request is not authorized to perform this operation using this permission.

I dont understand why azcopy throws permission mismatch even after logging in using azcopy login. No additional step should be required. I followed this [documantation](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10).

I am already owner of the storage account and have all the permissions.
![storage permission](/assets/hdi/storage-permission.jpg)

I was able to upload file using the storage account portal though. Hence I am authorized. Somehow I am not able to use azcopy.

The above document mentions using SAS token. Sol let's find a way to create a SAS token in my container. I had previous knowledge that SAS tokens are created at the storage account level. Hence I went to the storage account and created a SAS token.
![SAS](/assets/hdi/SAS.jpg)


azcopy copy sffiredata.csv "https://krishansparkhdistorage.file.core.windows.net/spark-container1/?**SAS_token**"

    Elapsed Time (Minutes): 0.0334
    Total Number Of Transfers: 1
    Number of Transfers Completed: 0
    Number of Transfers Failed: 1
    Final Job Status: Failed

This also didn't work. Now it's time to take pause. Next step will be to go through some HDI course before trying to use HDI again.

I might be doing a lot of things wrong here. I am not an expert in Spark and my experience with azure services are also limited to Azure ML only. I, however, believe for cloud providers like azure, the documentation should be straight forward for someone with working knowledge on spark to start using services like HDI and azure storage.

## Analysis using AWS EMR
[@swetashre](https://github.com/swetashre) from Amazon helped me run the analysis using AWS. AWS was relatively easier than Azure. Both Jupyter notebook and file upload to S3 were very easy and the spark queries ran very very fast.

[Instructions and code](https://github.com/krishansubudhi/sparkparactice/blob/master/SFFireDeptDataAnalysis_AWS.ipynb)