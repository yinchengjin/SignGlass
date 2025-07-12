import os

def strip_lines(list_lines):
    tmp_list_lines = []
    for line in list_lines:
        tmp_line = line.strip()
        tmp_list_lines.append(tmp_line)
    return tmp_list_lines

def get_user_file_index(user_file_index):
    int_list_user_file_index = []
    for file_index in user_file_index:
        int_list_user_file_index.append(int(file_index))
    return int_list_user_file_index

def get_file_name(list_file_name):
    int_list_file_name = []
    for file_name in list_file_name:
        # tmp_file_name_int = [int(x) for x in file_name.split(",")]
        tmp_file_name_int = [x for x in file_name.split(",")]
        # print(tmp_file_name_int)
        int_list_file_name.append(tmp_file_name_int[0])
    return int_list_file_name

def generate_file_name(list_of_folder_name,list_user_file_index,list_user_name, list_file_name,copy_version = 0):
    user_index = list_user_file_index[0]
    file_index = list_user_file_index[1]
    section_num = list_user_file_index[2]
    # print(user_index)
    # print(file_index)
    # print(section_num)
    # print(list_user_name)
    # print(list_file_name)
    str_copy_version = ""

    if copy_version != 0:
        str_copy_version = "_"+str(copy_version)
    tmp_file_name = list_user_name[user_index] + "_" + str(list_file_name[file_index])


    left_hand_imu_unsync = list_of_folder_name[2]  +tmp_file_name + "_section"+ str(section_num)+ "_left_unsync"+str_copy_version+".txt"
    right_hand_imu_unsync = list_of_folder_name[4] + tmp_file_name + "_section"+ str(section_num)+ "_right_unsync"+str_copy_version+".txt"

    ear_imu = list_of_folder_name[0] + tmp_file_name + "_section" + str(section_num) + "_ear"+str_copy_version+".txt"
    left_hand_imu = list_of_folder_name[1] + tmp_file_name + "_section" + str(section_num) + "_left"+str_copy_version+".txt"
    right_hand_imu = list_of_folder_name[3] + tmp_file_name + "_section" + str(section_num) + "_right"+str_copy_version+".txt"

    video_imu = list_of_folder_name[5] + tmp_file_name + "_section" + str(section_num) +str_copy_version+".mp4"

    imu_info = list_of_folder_name[6] + tmp_file_name + "_section" + str(section_num) + "_imu_info"+str_copy_version+".txt"
    return [ear_imu,left_hand_imu,left_hand_imu_unsync,right_hand_imu,right_hand_imu_unsync,video_imu,imu_info]


def get_user_name(parent_path,user_index):
    file_user_name = open(parent_path + "/file_name_tracking/user_name.txt", "r")
    list_user_name = strip_lines(file_user_name.readlines())
    return list_user_name[user_index]

def get_video_list_file(parent_path,user_name):
    video_list_path = parent_path+"/file_name_tracking/user_slides/"+user_name+"/all_selected_videos.txt"
    return video_list_path

def update_user_file_index(list_user_file_index,parent_path):

    user_index = list_user_file_index[0]
    file_index = list_user_file_index[1]
    section_index = list_user_file_index[2]

    file_user_file_index = open(parent_path+"/file_name_tracking/file_index.txt", "w")
    user_name = get_user_name(parent_path, user_index)
    file_list_name = open(get_video_list_file(parent_path,user_name), "r")
    size_of_file_list = len(file_list_name.readlines())

    if file_index == size_of_file_list:
        user_index = user_index + 1
        file_index = -1

    file_user_file_index.write(str(user_index)+"\n")
    file_user_file_index.write(str(file_index)+"\n")
    file_user_file_index.write(str(section_index ))
    file_user_file_index.close()


def create_folder(list_user_file_index,list_user_name,parent_path):

    user_result_path = parent_path+"/experiment result/local/"+list_user_name[list_user_file_index[0]]+"/"

    if not os.path.exists(user_result_path):
        os.makedirs(user_result_path)

    return user_result_path

def get_list_of_file_name(parent_path,redo_status=0,copy_version = 0):  # confirms that the code is under main function

    user_file_index = open(parent_path+"/file_name_tracking/file_index.txt", "r")
    list_user_file_index = get_user_file_index(strip_lines(user_file_index.readlines()))

    file_user_name = open(parent_path+"/file_name_tracking/user_name.txt", "r")
    list_user_name = strip_lines(file_user_name.readlines())

    user_index = list_user_file_index[0]
    user_name = get_user_name(parent_path, user_index)

    file_file_name = open(get_video_list_file(parent_path,user_name), "r")

    list_file_name = get_file_name(strip_lines(file_file_name.readlines()))
    user_index = list_user_file_index[0]

    if redo_status == 1:
        list_user_file_index[1] -=1


    print("\n\n"+str(list_user_file_index))
    # print(list_user_name)
    # print(list_file_name)

    # print("user_name:",list_user_name[list_user_file_index[0]])
    # print("sentence#:",list_file_name[list_user_file_index[1]])
    # print("secion#:",list_user_file_index[2])
    user_file_index.close()
    file_user_name.close()
    file_file_name.close()

    create_folder(list_user_file_index, list_user_name, parent_path)


    file_info = [list_user_name[list_user_file_index[0]],list_file_name[list_user_file_index[1]],list_user_file_index[2]]

    list_user_file_index[1] += 1
    update_user_file_index(list_user_file_index, parent_path)
    return file_info

if __name__ == "__main__":
    parent_path = "./hardware/signglass_raspi"
    file_list = get_list_of_file_name(parent_path)
    print(file_list)
    # get_list_of_file_name_cali()