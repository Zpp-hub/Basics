import os
import cv2
import json
import numpy as np
import numpy as np
"""
函数名称：ellipseToPolygon
函数功能：求椭圆上点的坐标
输入：
    a：椭圆的短半轴长
    b：椭圆的长半轴长
    xc：椭圆中心点x坐标
    yc：椭圆中心点y坐标
    n：生成椭圆坐标点个数
输出：
    splinePt：n个点的坐标
"""
def ellipseToPolygon(a, b, xc, yc, n=64):
    # ts = 0:np.pi / n:pi * 2
    ts = np.arange(0,2*np.pi,np.pi/(64/2))
    x = xc + a * np.cos(ts)
    y = yc + b * np.sin(ts)
    splinePt = np.zeros((len(y), 2))
    splinePt[:, 0] = x
    splinePt[:, 1] = y
    return [splinePt]
"""
函数名称：oval
函数功能：求椭圆基本信息
输入：
    points：四个顶点、或两个对角点
    n：椭圆上坐标点个数
输出：
    a：椭圆的短半轴长
    b：椭圆的长半轴长
    xc：椭圆中心点x坐标
    yc：椭圆中心点y坐标
    n：生成椭圆坐标点个数
"""
def oval(points,n=64):
    if (len(points) == 2):
         x1 = points[0][0]
         y1 = points[0][1]
         x2 = points[1][0]
         y2 = points[1][1]
    else:
        x1 = points[0][0]
        x2 = points[1][0]
        y1 = points[0][1]
        y2 = points[2][1]
    a = abs(np.diff([x1,x2])) / 2 # 横坐标相邻距离
    b = abs(np.diff([y1,y2])) / 2 # 纵坐标相邻距离
    xc = np.mean([x1,x2]) # 中心点
    yc = np.mean([y1,y2]) # 中心点
    return ellipseToPolygon(a, b, xc, yc,n)
"""
函数名称：polygon
函数功能：相对位置转化为绝对位置
输入：
    points：相对位置坐标
    width：图像的宽
    height：图像的高
输出：
    linePt：绝对位置点的坐标
"""
def polygon(points, width, height):
    linePt = []
    for k in range(len(points)):
        point = points[k]
        x = point['x']*width
        y = point['y']*height
        linePt.append([x, y])
    return linePt
def EvaluateCardinal2D(P0,P1,P2,P3,T,u):
    s= float(1-T)/2
    MC=np.array([[-s,2-s,s-2,s],[2.*s,s-3,3-(2.*s),-s],[-s,0,s,0],[0,1,0,0]],np.float)
    GHx= np.array([[P0[0]],[P1[0]],[P2[0]],[P3[0]]],np.float)
    GHy=np.array([[P0[1]],[P1[1]],[P2[1]],[P3[1]]],np.float)
    U=np.array([u**3,u**2,u,1],np.float)
    xt=np.dot(np.dot(U,MC),GHx)
    yt=np.dot(np.dot(U,MC),GHy)
    return xt,yt
def EvaluateCardinal2DAtNplusOneValues(P0,P1,P2,P3,T,N):
    xvec=[]
    yvec=[]
    u=0
    xve,yve =EvaluateCardinal2D(P0,P1,P2,P3,T,u)
    xvec.append(xve)
    yvec.append(yve)
    du=float(1)/N
    for k in range(N):
        u=k*du
        xve,yve =EvaluateCardinal2D(P0,P1,P2,P3,T,u)
        xvec.append(xve)
        yvec.append(yve)
    return xvec,yvec
def curve(PT,isin,Tension=0,n=100):
    setx = []
    sety = []
    Px = []
    Py = []
    for i in range(len(PT)):
        if(isin==False and i==0):
            Px.append(PT[0][0])
            Py.append(PT[0][1])
        Px.append(PT[i][0])
        Py.append(PT[i][1])


    if (isin):
        Px.append(PT[0][0])
        Py.append(PT[0][1])
        Px.append(PT[1][0])
        Py.append(PT[1][1])
        Px.append(PT[2][0])
        Py.append(PT[2][1])
    else:
        Px.append(PT[-1][0])
        Py.append(PT[-1][1])
    n = int(n/(len(Px)-3))
    for k in range(len(Px)-3):
        xvec, yvec = EvaluateCardinal2DAtNplusOneValues([Px[k], Py[k]], [Px[k + 1], Py[k + 1]], [Px[k + 2], Py[k + 2]],[Px[k + 3], Py[k + 3]], Tension, n)
        setx = setx+xvec
        sety = sety+yvec
    splinePt = np.zeros((len(sety),2))
    arange = np.arange(len(sety))
    splinePt[:, 0] = setx
    splinePt[:, 1] = sety
    # for k in range(len(setx)):
    #     splinePt.append([setx[k],sety[k]])
    return splinePt
def line_fun(point, x):
    if point[0][0] != 0:
        hight = int(point[1][1]) - int(point[0][1])
        weight = int(point[1][0]) - int(point[0][0])
        if weight == 0:
            weight = 1
        point.insert(0, [0, point[0][1] - (hight / weight) * point[0][0]])

    if point[len(point) - 1][0] != x:
        hight = int(point[len(point) - 1][1]) - int(point[len(point) - 2][1])
        weight = int(point[len(point) - 1][0]) - int(point[len(point) - 2][0])
        if weight == 0:
            weight = 1
        point.insert(len(point), [x, point[len(point) - 1][1] + (hight / weight) * (x - point[len(point) - 1][0])])
    return point


os.chdir('/home/zpp/zhihui')

dict_1 = {'above_internal_limiting_membrane': 25, 'below_retinal_nerve_fiber_layer': 50,
          'below_inner_plexiform_layer': 75, 'below_inner_muclear_layer': 100,
          'below_outer_plexiform_layer': 125, 'below_ellipsoid_zone': 150,
          'below_RPE': 175, 'below_choroid': 0}

list_layer = [['above_internal_limiting_membrane', 'below_retinal_nerve_fiber_layer'],
              ['below_retinal_nerve_fiber_layer', 'below_inner_plexiform_layer'],
              ['below_inner_plexiform_layer', 'below_inner_muclear_layer'],
              ['below_inner_muclear_layer', 'below_outer_plexiform_layer'],
              ['below_outer_plexiform_layer', 'below_ellipsoid_zone'],
              ['below_ellipsoid_zone', 'below_RPE'],
              ['below_RPE', 'below_choroid']]

dict_2 = ['above_internal_limiting_membrane', 'below_retinal_nerve_fiber_layer',
          'below_inner_plexiform_layer', 'below_inner_muclear_layer',
          'below_outer_plexiform_layer', 'below_ellipsoid_zone',
          'below_RPE', 'below_choroid']

kk = 0
for line in open('20190819_XieHe_GAN_AMD_OCT.csv', encoding='utf-8'):
    kk += 1
    if (kk == 1):
        continue
    line = line.replace('\n', '').split('"|"')
    json_d = line[6].replace('"{', '{').replace('}"', '}')
    name = line[2]
    frame_d = json.loads(json_d)

    annotations = frame_d['annotations']
    imageWidth = frame_d['imageWidth']
    imageHeight = frame_d['imageHeight']

    img = np.zeros([imageWidth, imageHeight, 3], np.uint8)
    layer = {}
    for i in range(len(annotations)):
        annotation = annotations[i]
        key = str(annotation['key'])
        shape = annotation['shape']
        points = shape['points']
        type = shape['type']
        biaopoints = []
        for k in range(len(points)):
            point = points[k]
            x = point['x'] * imageWidth
            y = point['y'] * imageHeight
            biaopoints.append([x, y])
        if biaopoints[0][0] > biaopoints[len(biaopoints)-1][0]:
            biaopoints = list(reversed(biaopoints))
        if (type == 'contour' and len(biaopoints) > 2):
            newPoints = curve(biaopoints, True, n=1000).tolist()
        elif (type == 'polygon'):
            newPoints = biaopoints
        elif (type == 'curve'):
            newPoints = curve(biaopoints, False, n=1000).tolist()
        else:
            print(type)

        if key in dict_1.keys():
            newPoints_1 = line_fun(newPoints, imageWidth)
            layer[key] = newPoints_1

        layer_ = [[i, dict_2.index(i)] for i in dict_2 if i in layer.keys()]

    for n in range(len(layer_) - 1):
        if layer_[n + 1][1] - layer_[n][1] == 1:
            col = (dict_1[layer_[n][0]], dict_1[layer_[n][0]], dict_1[layer_[n][0]])
            point2 = layer[layer_[n][0]] + list(reversed(layer[layer_[n + 1][0]]))
            point2 = [[round(i[0], 5), round(i[1], 5)] for i in point2]
            point3 = np.array(point2, dtype=np.int64)
            cv2.fillPoly(img, [point3], col)
        else:
            print(name)
            print(layer_[n + 1][1] - layer_[n][1])
            print(layer_)
            col = (255, 255, 255)
            point2 = layer[layer_[n][0]] + list(reversed(layer[layer_[n + 1][0]]))
            point3 = np.array(point2, dtype=np.int32)
            cv2.fillPoly(img, [point3], col)
    cv2.imwrite('layer' + '/' + name.replace('.jpg', '.png'), img)






