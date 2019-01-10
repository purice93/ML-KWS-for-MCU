
import  urllib.request
for j in range(18011210000,18011210600):
    url = r'http://xxcapp.xidian.edu.cn/excel/wap/project/index?search%5Bxuegonghao%5D='+str(j)+'&id=6'
    # url = r'http://xxcapp.xidian.edu.cn/excel/wap/project/index?search%5Bxuegonghao%5D=18031211408&id=6'
    res = urllib.request.urlopen(url)
    html = res.read().decode('utf-8')
    index = html.find('黄大勇')
    if j%10 ==0:
        print(j)
    if index!=-1:
        print('180'+str(j))