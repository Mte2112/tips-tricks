import os

# set variable for use
subdir = dir_ex
os.system('mkdir /newdir')
os.system('mkdir /newdir/%s'%subdir)
# output is new directory labeled /newdir/dir_ex

