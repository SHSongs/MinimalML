path = 'http://10.XXX.XX.XXX:8888/view/SCSWorkSpace/MinimalAI/AI/unet-regression/resIG/val/png/'
imgtypes = ['_label', '_input', '_output']
for i in range(0, 1000, 100):
    for imgtype in imgtypes:
        imgPath = path + format(i, '04') + imgtype + '.png'
        print(imgPath)
