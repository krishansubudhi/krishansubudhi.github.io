---
author: krishan
layout: post
categories: git
title: Undo a git rebase
---

Suppose you did a `git rebase` in your local branch but mistakenly rebased to an older branch and pushed changes to remote, then here is the solution to revert your changes and go back to the previous state.

1. Back up all your changes.
2. Use [git reflog](https://git-scm.com/docs/git-reflog) to see all your previous operations. `git log` will show rebased and squashed changes only.
3. Find out the commit where you want to go back to. Most probably this will be the commit before your rebase operation.
You will see commit ids like `HEAD@{16}`
4. Now reset your local branch to this commit.
    
    `git reset --hard HEAD@{16}`
      
  This only moves the local branch to HEAD@{16}

5. Now check the status of your local branch.
      
        git status

        On branch my_branch
        Your branch and 'origin/my_branch' have diverged

    Now this is frustrating.

    This means your local branch and remote branch have diverged as your local branch has gone back to a previous commit while remote branch is still ahead.

6. Force your remote branch to go back to where your local branch is.

        git push --force
    
NOTE:
At any stage, if you think you have made a mistake and want to reset your local branch with the current remote branch HEAD, 

      git reset --hard origin/HEAD
