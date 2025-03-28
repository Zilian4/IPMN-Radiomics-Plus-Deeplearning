import os 
dir_path = '/data/Ziliang/IPMN/IPMN_images_masks/t2/images'
for name in os.listdir(dir_path):
    if not name.startswith('IU'):
        if name.endswith('_1.nii.gz'):
            old_name = os.path.join(dir_path,name)
            print(old_name)
            name = name.split('_1.nii.gz')[0]
            new_name = os.path.join(dir_path,name+'.nii.gz')
            print(new_name)
            os.rename(old_name,new_name)