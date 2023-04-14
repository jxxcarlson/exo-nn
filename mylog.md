# Working with Python on Exosphere

This document is based on my experience using Exosphere's
compute resources, beginning with some easy exercises
to get me back in shape with Python and its use 
in machine learning.

## Set up the compute environment on exosphere

1. Use exosphere to create an instance (jxx-experiment) 
2. Connect to it using the terminal option
3. Create and cd to a working directory (nnet)
4. Set up and test a basic Python environment
   - $ `pip install pipenv --user``
   - $ `exit` # you need to log out and log back in for 
        the next step to work.
   - $ `pipenv shell`
   - $ `python`
   - ``` >>> 2 + 2
     ```
  - The reply is `4`, so we know that python is properly installed.
   
   - Install more packages
     - $ `pipenv install numpy`
     - $ `pipenv install matplotlib`
     - $ `pipenv install scikit-learn`
     - etc.


## Set up communication with Github

Here we assume that we already have some code on Github,
say in repository `git@github.com:jxxcarlson/exo-nn.git`

1. Create public/private key pair on exosphere
  - $ `ssh-keygen -t rsa`. This This will create a private key written to /`home/youruser/.ssh/id_rsa` and a public key written to `/home/youruser/.ssh/id_rsa.pub`.
  - Copy the contents of `.ssh/id_rsa.pub` and add it to your SSH keys on 
Github.  

2. Clone Github repo: `git pull git@github.com:jxxcarlson/exo-nn.git`.

3. Go to directory `./exo-nn`             
|

