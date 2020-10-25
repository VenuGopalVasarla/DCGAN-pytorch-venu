# LSUN dataloading issue with torchvision.
'Index Error' at line 109 in /pythonx.x/site-packages/torchvision/dataset/lsun.py

## Time of occurance:
 When a list of selected classes are provided to the 'torchvision.datasets.lsun()' method.

## Reason:
'verify_str_arg' is not valid function for classes argument when used with list type.

## Fix:
In order to fix this error caused by torchvision, please remove/comment the function inside the file  /pythonx.x/site-packages/torchvision/dataset/lsun.py
            
            #verify_str_arg(c, custom_msg=msg_fmtstr.format(type(c)))