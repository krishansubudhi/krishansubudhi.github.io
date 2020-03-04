---
comments: true
author: krishan
layout: post
categories: powershell
title: Powershell bashrc equivalent
---

Linux has a file called .bashrc which gets executed whenever a new terminal starts.
This .bashrc file is generally used for

1. Setting aliases
2. Setting up environment variables
3. Shortcut functions

In windows there is no .bashrc file. But there is a way to achieve the same functionality using powershell.

In this example we will download notepad++ and set an alias `vim` to open files in notepad++ from command prompt. 

`vim <filepath>' should open the file in notepad++

1. First download notepad++. Google search , download and install. 
2. Open powershell. Set an alias for notepad++. Here I am setting the alias `vim`
    
        set-alias vim  'C:\Program Files\Notepad++\notepad++.exe'

    This will only work for the current shell. A new shell will not have this alias.
3. Create `$profile` file = powershell equivalent of linux `.bashrc`.

        > $profile
        C:\Users\<username>\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1
        
        > echo "set-alias vim  'C:\Program Files\Notepad++\notepad++.exe'"

4. Open another powershell session as administrator.

        > Set-ExecutionPolicy RemoteSigned

    select A

5. close all open powershell shells. Open a brand new powershell shell.

        > vim $profile

6. This is the `.bashrc` equivalent. Add more shortcuts and start up commands here.