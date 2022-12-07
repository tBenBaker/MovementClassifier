import sys
sys.path.append("../../")
import urllib.request
import pathlib
import pandas as pd

video_dir = pathlib.Path("./DanceProj1/videos")
#video_dir.mkdir(exist_ok=True, parents=True)

errors = [] # hold the filenames of failures
#get ids from test_ids.csv
ids = pd.read_csv("test_ids.csv")['id'].values

for i in ids:
    # Print progress: "1 of 100 (2 errors)"
    #print(f"{ids.index(i) + 1} of {len(ids)} ({len(errors)} errors)", end="\r")
    url = "https://storage.googleapis.com/aist_plusplus_public/20121228/visualization_kpts3d/" + i + "_kpt3d.mp4"
    try:
        urllib.request.urlretrieve(url, video_dir / (i + ".mp4"))
    except Exception as e:
        errors.append(i)
        print(e)
        # print("Error when processing " + i + ": " + str(e))

print("Errors when processing: " + str(errors))