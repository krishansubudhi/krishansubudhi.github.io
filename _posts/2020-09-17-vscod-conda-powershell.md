---
comments: true
author: krishan
layout: post
categories: vscode
title: Automatically activate conda evironment in Powershell for VSCode
description: How to set up VSCode to automatically activate anaconda environment for powershell.
---


VSCode automatically links conda environments in the integrated terminal through the python extension. 

This is done through the following command which automatically triggers for any new environment
    
    conda activate myenv

The integrated terminal also calls this command every time a new terminal is created

        >C:/Users/<username>/AppData/Local/Continuum/miniconda3/Scripts/activate

This shell to conda integration is done by the vscode python integration. 
[Github source](https://github.com/microsoft/vscode-python)

# The problem
But there is one problem. I don't like windows cmd. Most of the heavy scripting is done using powershell. The python extension officially does not support powershell for activating conda environments

> *"Note: conda environments cannot be automatically activated in the integrated terminal if PowerShell is set as the integrated shell. See [Integrated terminal - Configuration](https://code.visualstudio.com/docs/editor/integrated-terminal#_configuration) for how to change the shell."*
    - [Using python environments in VSCode](https://code.visualstudio.com/docs/python/environments)

The reason can be found from their source code (probably) which is written in typescript. But considering my zero experience in type script I won't go much detail into that. For now we will see the workaround to activate conda environments automatically using powershell instead of cmd.

Conda will not be recognized if powershell is the integrated terminal. That's why I have been using cmd since a long time.

![Terminal before](/assets/anaconda_powershell_vscode/terminal_before.jpg)

# The solution
Out of the two commands mentioned at the very beginning, 

    C:/Users/<username>/AppData/Local/Continuum/miniconda3/Scripts/activate

this command activates conda and adds it to system path. For powershell, the command can be found in the powershell shortcut for anaconda which can be searched from windows start menu.

<img src="/assets/anaconda_powershell_vscode/anaconda_powershel_shortcut.jpg" width="400">

Properties of the shortcut contain the initial commands to set up anaconda in powershell.

Mine says this

    %windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& 'C:\Users\<username>\AppData\Local\Continuum\miniconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\Users\<username>\AppData\Local\Continuum\miniconda3' "

Now, in vscode , change default shell to powershell

    ctrl+shift+P -> Terminal: Select Default Shell -> powershell

Open settings 

    ctrl+shift+P -> Preferences: Open Settings(JSON)

Add this line at the bottom before }

    "terminal.integrated.shellArgs.windows": [
        "-ExecutionPolicy",
        "ByPass",
        "-NoExit",
        "-Command",
        "& C:\\Users\\<username>\\AppData\\Local\\Continuum\\miniconda3\\shell\\condabin\\conda-hook.ps1",
        ";conda activate 'C:\\Users\\<username>\\AppData\\Local\\Continuum\\miniconda3'"
    ]

Make modifications based on your anaconda path and activation command in the powershell shortcut.

![Terminal after](/assets/anaconda_powershell_vscode/terminal.jpg)

The command `conda activate myenv` will be automatically called by the python extension. 

## References:

https://stackoverflow.com/a/61402982/1513792