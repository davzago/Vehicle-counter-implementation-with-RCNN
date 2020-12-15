import urllib.request
from time import time, sleep




url = "https://video.autostrade.it/video-mp4_hq/dt9/957027a5-763d-43aa-8e5e-6986d25b5d0d-0.mp4"
url1 = "https://video.autostrade.it/video-mp4_hq/dt4/2539e22f-e967-46ff-8bb9-eb944301da99-23.mp4"
url2 = "https://video.autostrade.it/video-mp4_hq/fipili/d2cf6111-8007-4e08-a737-28f8612839b7-14.mp4"

count=1
urllib.request.urlretrieve(url2,"data/videoa14/video0Scandicci.mp4")

for i in range(4):
    sleep(120)
    urllib.request.urlretrieve(url2, "data/videoa14/video" + str(count) + "Scandicci.mp4")
    count+=1