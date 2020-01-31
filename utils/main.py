import os
import subprocess
import numpy as np
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

S3 = boto3.resource('s3')

def hello():
    print('hello skt')
    
def make_lst(df, path):
    with open(path, 'w') as fp:
        for i,(id,row) in enumerate(df.iterrows()):
            
            fp.write(str(i) + '\t')
            
            for label in row:
                fp.write(str(label) + '\t')
            
            fp.write(str(id))
            fp.write('\n')
            
def make_rec(img_folder, lst_file, resize=None):
    
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    cmd = ["python", os.path.join(dir_path, "im2rec.py"), os.path.abspath(lst_file), os.path.abspath(img_folder), '--recursive', '--pass-through', '--pack-label']
    
    if resize:
        cmd.extend(['--resize', str(resize)])
        
    subprocess.check_output(cmd)
                
#     try:
#         subprocess.check_output(cmd)
#     except subprocess.CalledProcessError as e:
#         print(e.output)   
        
def split_df(df, ratio):
    msk = np.random.rand(len(df)) < ratio
    train_df = df[msk]
    val_df = df[~msk]
    
    return train_df, val_df 

def make_paths_and_channels(root):
    
    if root[-1] != '/':
        root += '/'
    
    s3_train = root + 'train/'
    s3_val = root + 'val/'
    s3_output = root + 'output/'

    s3_train_channel = sagemaker.session.s3_input(s3_train, distribution='FullyReplicated', content_type='application/x-image', s3_data_type='S3Prefix')
    s3_val_channel = sagemaker.session.s3_input(s3_val, distribution='FullyReplicated', content_type='application/x-image', s3_data_type='S3Prefix')

    return s3_train, s3_val, s3_output, s3_train_channel, s3_val_channel

def s3_upload(file_path, s3_path):

    global S3

    file_name = file_path.split('/')[-1]
    data = open(file_path, "rb")
    
    s3_path = s3_path.lstrip('s3://')
    
    bucket = s3_path.split('/')[0]
    key = '/'.join(s3_path.split('/')[1:-1] + [file_name])
                       
    S3.Bucket(bucket).put_object(Key=key, Body=data)

def make_estimator(job_name, s3_output, input_mode='Pipe', train_instance_count=1, train_instance_type='ml.p2.xlarge', train_volume_size = 30, train_max_run=360000):   
    
    role = get_execution_role()
    sess = sagemaker.Session()
    training_image = get_image_uri(sess.boto_region_name, 'image-classification', repo_version="latest")
    
    estimator = sagemaker.estimator.Estimator(training_image,
                                         role, 
                                         train_instance_count=train_instance_count, 
                                         train_instance_type=train_instance_type,
                                         train_volume_size = train_volume_size,
                                         train_max_run = train_max_run,
                                         input_mode= input_mode,
                                         output_path=s3_output,
                                         sagemaker_session=sess,
                                         base_job_name=job_name)
    
    return estimator
    

    
