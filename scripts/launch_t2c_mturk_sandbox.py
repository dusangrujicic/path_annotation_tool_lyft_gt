import glob
import sys

sys.path.append("../annotation_tool")

from easyturk.interface import launch_t2c_path_anno
import json
import random

hits = []
img_root = "https://t2c-path.s3.eu-west-3.amazonaws.com/imgs/"
file_root = "https://t2c-path.s3.eu-west-3.amazonaws.com/files/"
data_root = "download/files/"

data_files = glob.glob(data_root+"*_video_data.json")
random.shuffle(data_files)

for data in data_files[:10]:
    command_data = json.load(open(data, "r"))
    command_name = data.split("/")[-1].replace("_video_data.json", "")
    #frame_data = json.load(open(data_root+"frame_"+command_name+"_data.json","r"))
    hits.append({
        "url": img_root+"frontal_"+command_name+".jpg",
        "top-down":  img_root+"top_down_"+command_name+".png",
        "command_data": command_data,
        "frame_data": file_root+"frame_"+command_name+"_data.json"
    })


hit_ids = launch_t2c_path_anno(hits, 0.25, 10, sandbox=True,
                          title="Draw the path that the car should follow. finals",
                          duration=1200, max_assignments=1, use_masters=False)

json.dump(hits, open("t2c_path.json", "w"))

with open("t2c_path_sandbox.json", "w") as f:
    json.dump(hit_ids, f)