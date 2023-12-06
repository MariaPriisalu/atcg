#!/usr/bin/env python
# coding: utf-8

# In[17]:


import glob
from distutils.dir_util import copy_tree
import os

outputBasePath = "./DatasetOut"
testIndex = 0


scenePattern = '[a-zA-Z0-9]'
for name in glob.glob('./*/*/[0-9]*/[0-9]*'):
    fromPath = name
    toPath = os.path.join(outputBasePath, f"test_{testIndex}")
    print(f"Copying from {fromPath} to {toPath}...")
    copy_tree(fromPath, toPath)
    testIndex += 1
    #break
    
print("Finished")
    


# In[ ]:




