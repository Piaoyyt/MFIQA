import os,sys

if sys.platform=="linux":
    index = os.getcwd().split("/").index("My_project")
    project_path = '/'.join(os.getcwd().split("/")[:index+1])
else:
    index = os.getcwd().split("\\").index("My_project")
    project_path = '\\'.join(os.getcwd().split("\\")[:index+1])

# print(project_path)