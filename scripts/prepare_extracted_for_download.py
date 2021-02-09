import pathlib
import os
import tqdm
import shutil
pathlib.Path("download/imgs").mkdir(parents=True, exist_ok=True)
pathlib.Path("download/files").mkdir(parents=True, exist_ok=True)
to_copy = [x for x in os.walk("../extracted_data/")][1:]

for (pth, _, files) in tqdm.tqdm(to_copy):
    assert len(files) == 4
    for file in files:
        if file.split(".")[-1] in ["png", "jpg"]:
            shutil.copyfile(pth+"/"+file, "download/imgs/"+file)
        elif file.split(".")[-1] in ["json"]:
            if file == "video_data.json":
                shutil.copyfile(pth+"/" + file, "download/files/" + pth.split("/")[-1]+"_"+file)
            else:
                shutil.copyfile(pth+"/" + file, "download/files/" + file)
        else:
            raise Exception('file extension', file.split(".")[-1], "not known")
