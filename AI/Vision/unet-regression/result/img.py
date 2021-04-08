import cv2
import os
import numpy as np

data_dir = 'ramdom'

lst_data = os.listdir(data_dir)

lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('png')]
lst_data.sort()


def list_chunk(lst, n):
    return [lst[i: i + n] for i in range(0, len(lst), n)]


lst_data = list_chunk(lst_data, 3)

full = []
for val_set in lst_data:
    hmlst = []

    for i in [1, 0, 2]:
        s = cv2.imread(os.path.join(data_dir, val_set[i]))
        hmlst.append(s)

        print(lst_data[0][i])

    addh = cv2.hconcat(hmlst)
    text = val_set[0][0:4]
    sz = addh.shape
    org = (sz[1] - 100, sz[0] - 250)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(addh, text, org, font, 1, (255, 255, 0), 5)

    full.append(addh)

addv = cv2.vconcat(full)
cv2.imshow('', addv)
cv2.imwrite('random.png', addv)
cv2.waitKey(0)
cv2.destroyAllWindows()
