import os

main_folder = r"D:\LiGD\NoahDF\arcDataset"
sub_folders = [name for name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, name))]

for folder in sub_folders:
    temp_path = os.path.join(main_folder,folder)
    pic_list = os.listdir(temp_path)
    num = len(pic_list)
    for n in range(num):
        old_dir = os.path.join(main_folder,folder,pic_list[n])
        temp_name_list = [str(n+1),"jpg"]
        seperator = '.'
        new_name = seperator.join(temp_name_list)
        new_dir = os.path.join(main_folder,folder,new_name)
        if old_dir:
            os.rename(old_dir,new_dir)

