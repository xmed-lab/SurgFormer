import os
import numpy as np
import pickle

# root_dir = '/research/dept6/yhlong/Miccai19_challenge/Miccai19/'
# img_dir = os.path.join(root_dir, 'frame')
# phase_dir = os.path.join(root_dir, 'Annotation', 'phase')
# tool_dir = os.path.join(root_dir, 'Annotation', 'tool')

root_dir2 = '/nfs/usrhome/eemenglan/71_heart/'

img_dir2 = os.path.join(root_dir2, 'phase_frames')
phase_dir2 = os.path.join(root_dir2, 'phase_annotations')
tool_dir2 = os.path.join(root_dir2, 'instrument_annotations')
phase_ant_dir2 = os.path.join(root_dir2, 'phase_anticipation_annotations')
tool_ant_dir2 = os.path.join(root_dir2, 'instrument_anti_annotations')

print(root_dir2)
print(img_dir2)
print(phase_dir2)
print(phase_ant_dir2)
print(tool_ant_dir2)

# cholec80==================
def get_dirs2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort(key=lambda x: int(x))
    file_paths.sort(key=lambda x: int(os.path.basename(x)))
    return file_names, file_paths


def get_files2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths


img_dir_names2, img_dir_paths2 = get_dirs2(img_dir2)
tool_file_names2, tool_file_paths2 = get_files2(tool_dir2)
phase_file_names2, phase_file_paths2 = get_files2(phase_dir2)
phase_ant_file_names2, phase_ant_file_paths2 = get_files2(phase_ant_dir2)
tool_ant_file_names2, tool_ant_file_paths2 = get_files2(tool_ant_dir2)

phase_dict = {}
phase_dict_key = ['Preparation','SuspendPericardium', 'DissociateVein', 'SuturePerfusionNeedleSpacer', 'InsertPerfusionNeedle', 'BlockAorta', 'LeftHeartDrain',
                          'DissectLeftAtrium', 'ExposeMitralValve','MitralValvuloplasty', 'SuturetLeftAtrium', 'HemostasisPericardiumSuture']

tool_label = ['Needle_holder', 'Aspirator', 'Endotherm_knife', 'Needle_holder_small', 'Occlusion_forceps', 'Knife', 'Atrial_retractor', 'Scissor']

    # ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
    #               'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']

for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i
print(phase_dict)
# cholec80==================


# cholec80==================
all_info_all2 = []

for j in range(len(phase_file_names2)):
    downsample_rate = 1
    phase_file = open(phase_file_paths2[j])
    tool_file = open(tool_file_paths2[j])
    phase_ant_file = open(phase_ant_file_paths2[j])
    tool_ant_file = open(tool_ant_file_paths2[j])

    # video_num_file = int(os.path.splitext(os.path.basename(phase_file_paths2[j]))[0][5:7])
    video_num_dir = int(os.path.basename(img_dir_paths2[j]))

    print("video_num_dir:", video_num_dir, "rate:", downsample_rate)

    info_all = []
    first_line = True
    for phase_line in phase_file:
        phase_split = phase_line.split()
        if first_line:
            first_line = False
            continue
        if int(phase_split[0]) % downsample_rate == 0:
            info_each = []
            img_file_each_path = os.path.join(img_dir_paths2[j], phase_split[0] + '.jpg')
            info_each.append(img_file_each_path)
            info_each.append(phase_dict[phase_split[1]])
            info_all.append(info_each)  # 2 items


    count_tool = 0
    first_line = True
    for tool_line in tool_file:
        tool_split = tool_line.split()
        if first_line:
            first_line = False
            continue
        if int(tool_split[0]) % downsample_rate == 0:
            info_all[count_tool].append(int(tool_split[0 + 1]))
            info_all[count_tool].append(int(tool_split[1 + 1]))
            info_all[count_tool].append(int(tool_split[2 + 1]))
            info_all[count_tool].append(int(tool_split[3 + 1]))
            info_all[count_tool].append(int(tool_split[4 + 1]))
            info_all[count_tool].append(int(tool_split[5 + 1]))
            info_all[count_tool].append(int(tool_split[6 + 1]))
            info_all[count_tool].append(int(tool_split[7 + 1])) # 8 items
            # info_all[count_tool].append(int(0))
            count_tool += 1

    count_phase_ant = 0
    for phase_ant_line in phase_ant_file:
        phase_ant_split = phase_ant_line.split()
        info_all[count_phase_ant].append(float(phase_ant_split[0]))
        info_all[count_phase_ant].append(float(phase_ant_split[1]))
        info_all[count_phase_ant].append(float(phase_ant_split[2]))
        info_all[count_phase_ant].append(float(phase_ant_split[3]))
        info_all[count_phase_ant].append(float(phase_ant_split[4]))
        info_all[count_phase_ant].append(float(phase_ant_split[5]))
        info_all[count_phase_ant].append(float(phase_ant_split[6]))
        info_all[count_phase_ant].append(float(phase_ant_split[7]))
        info_all[count_phase_ant].append(float(phase_ant_split[8]))
        info_all[count_phase_ant].append(float(phase_ant_split[9]))
        info_all[count_phase_ant].append(float(phase_ant_split[10]))
        info_all[count_phase_ant].append(float(phase_ant_split[11])) # 12 items
        count_phase_ant += 1

    count_tool_ant = 0
    for tool_ant_line in tool_ant_file:
        tool_ant_split = tool_ant_line.split()
        info_all[count_tool_ant].append(float(tool_ant_split[0]))
        info_all[count_tool_ant].append(float(tool_ant_split[1]))
        info_all[count_tool_ant].append(float(tool_ant_split[2]))
        info_all[count_tool_ant].append(float(tool_ant_split[3]))
        info_all[count_tool_ant].append(float(tool_ant_split[4]))
        info_all[count_tool_ant].append(float(tool_ant_split[5]))
        info_all[count_tool_ant].append(float(tool_ant_split[6]))
        info_all[count_tool_ant].append(float(tool_ant_split[7]))   # 8 items
        count_tool_ant += 1

    all_info_all2.append(info_all)


with open('./mvp_phase_tool_phase_ant_tool_ant_57.pkl', 'wb') as f:
    pickle.dump(all_info_all2, f)

video_nums = len(all_info_all2)
print('total number of videos', video_nums)
print('Done')
