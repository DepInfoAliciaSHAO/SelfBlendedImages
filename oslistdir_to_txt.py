import os

def save_listdir_txt(txt_name, dir_path):
    list_dir = os.listdir(dir_path)
    with open(txt_name, 'w') as f:
        for item in list_dir:
            f.write(item + '\n')

if __name__ == "__main__":
    VID_EXTENSIONS = {'.mp4', '.mov', '.avi'}
    folder_path = os.path.join('/home/alicia/dataShareID', "ShareIDTeam")
    folder_list = [os.path.join(folder_path, video) for video in os.listdir(folder_path) if os.path.splitext(video)[1].lower() in VID_EXTENSIONS]
    TXT_NAME = 'team_shareID.txt'
    # DIR_PATH = '/mnt/ssd_nvme2/datasets/FaceForensics++/sbi/frames/'
    # save_listdir_txt(TXT_NAME, DIR_PATH)
    with open(TXT_NAME, 'w') as f:
        for item in folder_list:
            f.write(item + '\n')