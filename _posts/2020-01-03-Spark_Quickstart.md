---
author: krishan
layout: post
description: How to install and run spark in local windows 10 machines using two installation methods - binaries and pip.
categories: deeplearning
title: Spark Quickstart on Windows 10 Machine
---

Apache Spark™ is a unified analytics engine for large-scale data processing.

1. [Install java](https://www.java.com/en/download/win10.jsp)
2. Install spark (2 ways)

    1. Using pyspark (trimmed down version of spark with only python binaries).
        
        *spark can also be run using java, scala, R and SQL using method b(pyspark only supports python)*

        ```
        conda create -n "spark"
        pip install pyspark
        ```

    2. Using spark binaries

        1.  download [spark binaries](http://spark.apache.org/downloads.html)

            ![download spark 2.4](/assets/spark-quickstart/download_spark.jpg)
            

        2. Install 7zip for winodows 64 bit and run this in power shell.

        ```
        > & 'C:\Program Files\7-Zip\7z.exe' -x .\spark-2.4.4-bin-hadoop2.7.tgz
        
        > & 'C:\Program Files\7-Zip\7z.exe' x .\spark-2.4.4-bin-hadoop2.7.tar

        PS C:\Users\krkusuk.REDMOND\bin> dir .\spark-2.4.4-bin-hadoop2.7\

        Directory: C:\Users\krkusuk.REDMOND\bin\spark-2.4.4-bin-hadoop2.7
        ```
        ```

            Mode                LastWriteTime         Length Name
            ----                -------------         ------ ----
            d-----        8/27/2019   2:30 PM                bin
            d-----        8/27/2019   2:30 PM                conf
            d-----        8/27/2019   2:30 PM                data
            d-----        8/27/2019   2:30 PM                examples
            d-----        8/27/2019   2:30 PM                jars
            d-----        8/27/2019   2:30 PM                kubernetes
            d-----        8/27/2019   2:30 PM                licenses
            d-----        8/27/2019   2:30 PM                python
            d-----        8/27/2019   2:30 PM                R
            d-----        8/27/2019   2:30 PM                sbin
            d-----        8/27/2019   2:30 PM                yarn
            -a----        8/27/2019   2:30 PM          21316 LICENSE
            -a----        8/27/2019   2:30 PM          42919 NOTICE
            -a----        8/27/2019   2:30 PM           3952 README.md
            -a----        8/27/2019   2:30 PM            164 RELEASE

        ```
    3. Add bin directory to PATHC:\Users\krkusuk.REDMOND\bin\spark-2.4.4-bin-hadoop2.7\bin
    
    
    
3. Test spark installation

    ```
    > pyspark
            ____              __
            / __/__  ___ _____/ /__
        _\ \/ _ \/ _ `/ __/  '_/
        /__ / .__/\_,_/_/ /_/\_\   version 2.4.4
            /_/

    Using Python version 3.7.4 (default, Aug  9 2019 18:34:13)
    SparkSession available as 'spark'.
    >>> spark.version
    '2.4.4'

    ```


4. Winutils error fix : https://stackoverflow.com/questions/35652665/java-io-ioexception-could-not-locate-executable-null-bin-winutils-exe-in-the-ha

5. Download CSV data in zip format from [usda employment website](https://www.ers.usda.gov/data-products/atlas-of-rural-and-small-town-america/download-the-data/).

    Unzip the file

    Play with spark shell

    ![scala and python spark shells](/assets/spark-quickstart/shells.png)
    
7. Run a full spark program in python

    1. If spark is installed through binary download
        
        Set your environment variable SPARK_HOME to root level directory where you installed Spark on your local machine.
        
        ![set spark home](/assets/spark-quickstart/spark_home_env.png)

        To avoid verbose INFO messages printed on the console, set rootCategory=WARN in the conf/ log4j.properties file.
        
        Rename log4j.properties.template to log4j.properties
        
            log4j.rootCategory=WARN, console

        
        Download [emp_analysis.py](https://github.com/krishansubudhi/sparkparactice/blob/master/emp_analysis.py) script.

        ```
        (base) PS C:\Users\krkusuk.REDMOND\Study\spark> spark-submit .\emp_analysis.py .\Rural_Atlas_Update20\Jobs.csv
        
        Results
        
        +-----+--------+
        |State|Counties|
        +-----+--------+
        |TX   |255     |
        |GA   |160     |
        |VA   |135     |
        |KY   |121     |
        |MO   |116     |
        |KS   |106     |
        |IL   |103     |
        |NC   |101     |
        |IA   |100     |
        |TN   |96      |
        +-----+--------+
        only showing top 10 rows
        
        Unemployement per states
        +-----+------------------+
        |State|AvgUnempRate2018  |
        +-----+------------------+
        |PR   |11.106329113924046|
        |AK   |8.616666666666665 |
        |AZ   |6.59375           |
        |WV   |5.864285714285716 |
        |DC   |5.6               |
        |WA   |5.597499999999999 |
        |LA   |5.583076923076922 |
        |MS   |5.540963855421685 |
        |NM   |5.341176470588236 |
        |CA   |5.222033898305086 |
        +-----+------------------+
        only showing top 10 rows

        Unemployment in Washington state
        +-----+-----------------+
        |State| AvgUnempRate2018|
        +-----+-----------------+
        |   WA|5.597499999999999|
        +-----+-----------------+

        ```


    8. If installed spark through pyspark.

        Check your spark installation directory in anaconda powershell.

        ```
        (spark) PS C:\Users\krkusuk> gcm pyspark

        CommandType     Name                                               Version    Source
        -----------     ----                                               -------    ------
        Application     pyspark.cmd                                        0.0.0.0

        C:\Users\krkusuk\AppData\Local\Continuum\miniconda3\envs\spark\Scripts\pyspark.cmd

        (spark) PS C:\Users\krkusuk> gcm spark-submit

        CommandType     Name                                               Version    Source
        -----------     ----                                               -------    ------
        Application     spark-submit.cmd                                   0.0.0.0  
        
        C:\Users\krkusuk\AppData\Local\Continuum\miniconda3\envs\spark\Scripts\spark-submit.cmd
        ```

        Problem with setting loglevel in pyspark
        2 solutions that worked for me.

        1. Through config file

            Get spark home

            ```
            >>> import os
            >>> os.environ["SPARK_HOME"]
            'C:\\Users\\krkusuk\\AppData\\Local\\Continuum\\miniconda3\\envs\\spark\\lib\\site-packages\\pyspark'
            ```
            Create conf folder and log4j.properties file inside conf folder.
            Write these inside the file.
            ```
            # Set everything to be logged to the console
            log4j.rootCategory=WARN, console
            log4j.appender.console=org.apache.log4j.ConsoleAppender
            log4j.appender.console.target=System.err
            log4j.appender.console.layout=org.apache.log4j.PatternLayout
            log4j.appender.console.layout.ConversionPattern=%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n
            ```
        2. Through code

            ```python
            spark = (SparkSession
                    .builder
                    .appName('PythonEmpAnalysis')
                    .getOrCreate())
            spark.sparkContext.setLogLevel('ERROR')
            ```
 

        Run

            spark-submit.cmd .\emp_analysis.py .\Rural_Atlas_Update20\Jobs.csv 
