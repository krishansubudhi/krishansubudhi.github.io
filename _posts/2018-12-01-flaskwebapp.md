---
author: krishan
layout: post
categories: webapp
title: Python Flask web application in azure linux 
---

Even though the tutorial involves azure, the instructions will work in any ubuntu based linux machine.

> For aws, there change port by visiting the security group of your vm. If you have selected generic amazon ami, which is most probably a cent-os flavour of linux, replace apt-get commands with yum.

## Prerequisites

	- Azure account
	- Azure subscription, resource group and a linux vm
	- python 3.6, virtualenv, flask
	- nginx, gunicorn

## Setting up the VM
Boot a linux VM in azure.
In the portal, navigate to the VM page, go to networking.
Add inbound rule for port 22 and 5000 in the security group of the VM. 

Port 22 will be used for ssh and 5000 for testing flask application.

### SSH to VM instance
	> ssh user@host

### Install dependencies
	$ sudo apt-get install python python-pip nginx

### Create virtualenv



	$ mkdir -p www/flaskapp
	$ cd www/flaskapp
	$ pip3 install virtualenv	
	$ python3 -m virtualenv flaskenv 	
	$ chmod 755 flaskenv/

> For cent-os replace python3 with python36 and pip3 with pip-3.6 . If you still see error, google how to install python 3.6 and pip 3.6 in cent-os.

### Install python dependencies

	$ source flaskenv/bin/activate
	$ python -V
	Python 3.6.7
	$ pip install flask

## Create a simple flask web application
	vim app.py

````python

from flask import Flask
app = Flask(__name__)
@app.route("/")
def helloworld():
	return 'Hello World!'
````
	
### Run the flask application
	$ FLASK_APP=app.py
	$ flask run

Open another ssh client, connect to the linux machine and run this command.
	
	$ curl localhost:5000

You should get 
	
	Hello World!

## Access flask application from your browser
By default ports  are not accessible from hosts other than local host.
Try accessing _http://\<server-IP\>:5000_ from local browser. You should not be able to access the page yet. The flask app is configured to accept connections from localhost only.
[This post](http://dixu.me/2015/10/26/How_to_Allow_Remote_Connections_to_Flask_Web_Service/) helped me understand the concept.



#### Restart the flask server allowing inbound connection from all ips
	$ flask run --host 0.0.0.0

### Check if your server is accessible from your laptop.
If your laptop is running windows, enable telnet for windows ( google the instructions ).
Open powershell and check if port 5000 is accessible
	
	telnet <server-IP> 5000

You should get a blank screen with blinking cursor. If not, port 5000 is not accessible from your laptop and you should go back to azure portal and enable inbound connection for your host.

### Enable TCP traffic through port 5000
Now go to the server shell and run

	$ sudo iptables -I INPUT -p tcp --dport 5000 -j ACCEPT

Open your browser and enter the url _http://\<server-IP\>:5000_

You should be able to see the response now.
**Hello World!**

## WSGI server and Web server
When you execute flask run , a warning message appears on the screen
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
Flask has a simple web server inbuilt which is suitable for small numeber of requests mostly for development purpose.
For production, we need a more powerful web server like apache or nginx.

### Web server: 
In layman's term web server is the program that handles client HTTP requests.
A web server has multiple functionalities too. It can handle cucurrency, applies logging, error handling and also acts as a reverse-proxy.
It can internally call other programs which are written in python/ruby/java with the request and serve the response from those programs. The client is unaware of the internals of how the request is seved.
Apache and nginx are examples of web servers.

[nginx explanation](https://www.fullstackpython.com/nginx.html)

### WSGI server:
In layman's term it is a broker between web servers and python programs.
A Web Server Gateway Interface (WSGI) server implements the web server side of the WSGI interface for running Python web applications. 
Python programs do not need to worry about connection to web servers. They only need to be WSGI compliant.
gunicorn is a WSGI server

[wsgi-servers explanation](https://www.fullstackpython.com/wsgi-servers.html)

### Web frameworks: 
A web framework contains tools and libraries to develop a web application.
A WSGI compliant web framework follow the WSGI standard.

Examples of web frameworks for python include: Django, Flask, Bottle, Pyramid

[Web framework explanation](https://en.wikipedia.org/wiki/Web_framework)

This image explains the flow quiet well.
![alt text](https://files.realpython.com/media/flask-nginx-gunicorn-architecture.012eb1c10f5e.jpg "Image explaining the flow")
[Picture source](https://realpython.com/kickstarting-flask-on-ubuntu-setup-and-deployment/)

## Configure WSGI server
### Attach gunicorn with flask application
	$ gunicorn --bind 0.0.0.0:5000 app:app --reload & >> /dev/null

Now your web page is served by gunicorn.To verify that gunicorn is working, open your browser and access _http://\<server-IP\>:5000_ again.

You should be able to see __Hello World!__

### Make changes to code
	$ vim app.py

change `return 'Hello World!'` to `return 'Hello World Again!'`

Now reload the page in your browser. It should display __ Hello World Again!__.

Notice you did not have to restart gunicorn. The flag `--reload` configures gunicorn to restart the application everytime there is a code change.

## Configure nginx web server
..Wait. Another web server!
If gunicorn is a poserful web server then, why another web server on top of it?
Well I can only guess that, nginx is very generic. It will help us host multiple web applications in multiple languages. While gunicorn is a WSGI server only.
There can be other reasons but I am not aware of them. For the sake of completion let's configure nginx too.

### Create nginx config file

	$ sudo rm /etc/nginx/sites-available/default
	$ vim /etc/nginx/sites-available/flaskpp

> For cent-os , you might have to create two folders in /etc/nginx and make changes to nginx.conf.
[More info here](https://stackoverflow.com/questions/17413526/nginx-missing-sites-available-directory)

Replace file content with

	server{
		listen 80;
		location / {
			proxy_pass http://localhost:5000;
		}
	}

Save the file.

Here we are configuring nginx to call gunicorn server for all incoming requests.

### Create a softlink of configuration in sites-enables folder.
	$sudo ln /etc/nginx/sites-available/flaskapp /etc//nginx/sites-enabled/flaskapp

### Restart nginx
	$ sudo /etc/init.d/nginx restart
	
nginx listens requests to port 80 which is the default http protocol. Verify that nginx is talking to gunicorn with thia command.

	$ curl localhost:80
	Hello World Again!

Perfect. nginx is now talking to gunicorn.

> If you see something else other then __Hello World Again!__ , then check if the default server config in nginx is deleted properly. In cent-os the config is in /etc/nginx/nginx.conf



Now let's try to access the website from the browser. But this time we will access through port 80 instead of 5000

_http://\<server-IP\>/_

No need to mention _:\<port\>_ for port 80

Now what happened ? The site is not available.
Well what changed? 

...

The answer is port. Is port 80 accessible from my laptop? How do I check that?

Same thing we used previously.

	> telnet <server-IP> 80
	

If this is coming in the console, that mean 80 is not open to public.

    Connecting To \<server-IP\>...

### Enable inbound to port 80 in azure
Go to azure portal and enable inbound from all ips to port 80.
Now go to your browser and go to _http://\<server-IP\>/_

Voila! You are about to connect to your web server.

## Summary
In this article, we explored

1. How to create a flask web application
2. Dependencies required
3. Types of web servers and their roles
4. How to check for port access using telnet
5. How to check if web application is accessible using curl and browser.
6. How to make ports public
7. How to start flask application through gunicorn
8. How to configure nginx 
9. How to run a fully fledged python web server in azure

Hope you liked this. Thanks.