import sklearn.pipeline as Pipeline
import importlib
import ast
from sklearn.ensemble import VotingClassifier

def make_pipeline_str(pipeline_str,verbose=1):
    if pipeline_str is None:
        return None

    list_pip_string=split_methods_pip(pipeline_str)
    list_pip_methods=[]

    for method_str in list_pip_string:
        method_str=method_str.split("(",1)

        method_call_str=method_str[0].split("/")

        attr_list_str=[]
        attr_list_str=split_attr(method_str[1][0:len(method_str[1])-1])

        try:
           imported_lib=importlib.import_module(method_call_str[0])
           kwargs_method={}
  
           for attr in attr_list_str:
               attr=attr.split("=",1)

               if attr[1].find("/")!=-1:
                   if attr[1].find("(") != -1:
                        pip_attr=make_pipeline_str(attr[1])
                        if pip_attr is None:
                            return None
                        kwargs_method[attr[0].strip()]=pip_attr
                   else:
                       attr_imported=attr[1].split("/")
                       imported_lib_attr=importlib.import_module(attr_imported[0])
                       attr_imported=getattr(imported_lib_attr,attr_imported[1])
                       kwargs_method[attr[0].strip()]=attr_imported

               else:
                   kwargs_method[attr[0].strip()]=ast.literal_eval(attr[1])

           method=getattr(imported_lib,method_call_str[1])(**kwargs_method)

           if hasattr(method, 'n_jobs'):
               method.n_jobs=1

           if hasattr(method, 'nthread'):
               method.nthread=1

           list_pip_methods.append(method)
        except  Exception as e:
           if verbose>0:
               print("Load method error: "+str(method_str))
               print(str(e))
           return None

    #If there is justs one method, return it outside a pipeline (RFE methods doesn't work otherwise)
    if len(list_pip_methods)==1:
        return list_pip_methods[0]

    try:
        pipeline=Pipeline.make_pipeline(*list_pip_methods)
    except Exception as e:
        if verbose>0:
            print("Pipeline definition error: "+str(list_pip_string))
            print(str(e))
        return None

    return pipeline



def split_attr(string_attr):
    split_string=[]
    count_nested=0
    index_start=0
    index_end=0
    for index_end,c in enumerate(string_attr):
        if c in ['(','[']:
            count_nested+=1
        if c in [')',']']:
            count_nested-=1

        if c==',' and count_nested==0:
            split_string.append(string_attr[index_start:index_end])
            index_start=index_end+1

        if count_nested<0:
            index_end-=1
            break
    
    if(string_attr.strip()!=""):
        split_string.append(string_attr[index_start:index_end+1])

    return split_string

def split_methods_pip(string_pip):
    split_string=[]
    count_nested=0
    index_start=0
    index_end=0
    for index_end,c in enumerate(string_pip):
        if c in ['(','[']:
            count_nested+=1
        if c in [')',']']:
            count_nested-=1

        if string_pip[index_end]=='-' and string_pip[index_end+1]=='>' and count_nested==0:
            split_string.append(string_pip[index_start:index_end])
            index_start=index_end+2
        
        if count_nested<0:
            index_end-=1
            break
    
    split_string.append(string_pip[index_start:index_end+1])

    return split_string
    

def make_voting_ensemble(pipelines_population):
    if pipelines_population is None:
        return None

    pipelines_population=pipelines_population.split("|")
    pipeline_list=[]

    for id_pip, pipeline_str in enumerate(pipelines_population):
        pipeline=make_pipeline_str(pipeline_str)
        if pipeline is not None:
            pipeline_list.append(("Pipe_"+str(id_pip),pipeline))

    return VotingClassifier(estimators=pipeline_list)       




