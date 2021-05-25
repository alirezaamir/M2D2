import numpy as np
import json


data = json.load(open("../../output/json/AttemptsResults_ccnn_unseen.json"))


for t_duration in [0, 4, 8, 15, 37, 75]:
    for num in range(1, 4):
        cnt = 0
        for pat in data.keys():
            for attempt in range(num):
                if data[pat]['{}_{}'.format(t_duration, attempt)] <= t_duration:
                    cnt += 1
                    break

        print("T : {}, #: {}, {}".format(t_duration, num, cnt))
