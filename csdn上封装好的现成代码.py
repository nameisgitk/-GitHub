#详见CSDN网页 https://blog.csdn.net/qq_56350439/article/details/121792540
from tkinter import *
from tkinter import messagebox
import cv2
import sys
import numpy as np
import os
from PIL import Image
import numpy as np
# 创建窗口：实例化一个窗口对象。
root = Tk()
#窗口大小
 
root.geometry("600x450+374+182")
#x代表*  宽乘高
#窗口标题
root.title("人脸识别")
# #添加标签控件
label=Label(root,text="录入数据按三次空格"
                      " 录入完成按1 识别完成按3",font=("楷体",20),fg="red")
#定位
label.grid(row=0,column=0)
# 添加输入框并定位
#放在第一行第二列的位置
def func1():
    def mkdir(path):
        # 判断目录是否存在
        # 存在：True
        # 不存在：False
        folder = os.path.exists(path)
 
        # 判断结果
        if not folder:
            # 如果不存在，则创建新目录
            os.makedirs(path)
            print('-----创建成功-----')
 
        else:
            # 如果目录已存在，则不创建，提示目录已存在
            print(path + '目录已存在')
 
    path = "D:/information/"
    mkdir(path)
    cap=cv2.VideoCapture(0)
    flag=1
    num=1
    while(cap.isOpened()):#检测摄像头是否开启
        ret_flag,Vshow=cap.read()#得到每帧图像
        cv2.imshow("Capture_Test",Vshow)#显示图像
        k=cv2.waitKey(1) & 0xFF#压下按键判断
        if k==ord(' '):
            cv2.imwrite("D:/information/"+str(num)+"."+str(entry.get())+".jpg",Vshow)#把这个图像压倒这个路径里面
            print("success to save"+str(num)+"jpg")#保存成功
            print("------------")
            num+=1
        elif k==ord('1'):
            break;
#释放摄像头
    cap.release()
#释放内存
    cv2.destroyAllWindows()
def func2():
    def mkdir(path):
        # 判断目录是否存在
        # 存在：True
        # 不存在：False
        folder = os.path.exists(path)
 
        # 判断结果
        if not folder:
            # 如果不存在，则创建新目录
            os.makedirs(path)
            print('-----创建成功-----')
 
        else:
            # 如果目录已存在，则不创建，提示目录已存在
            print(path + '目录已存在')
 
    path = "D:/trainer/"
    mkdir(path)
    def getImageAndLabels(path):
        # 保存人脸数据
        facesSamples = []
        # 保存姓名数据
        ids = []
        # 保存图片信息
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        # 加载分类器
        face_detector = cv2.CascadeClassifier(
            "D:\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml")
        # 遍历列表中的图片
        for imagePath in imagePaths:
            # 打开图片,灰度化PIL有九种不同格式
            PIL_img = Image.open(imagePath).convert('L')
            # 将图像转换为数组，以黑白深浅
            img_numpy = np.array(PIL_img, 'uint8')
            # 获取图片人脸特征
            faces = face_detector.detectMultiScale(img_numpy)
            # 获取每张图片的id和姓名
            id = int(os.path.split(imagePath)[1].split('.')[0])
            # 预防无面容照片
            for x, y, w, h in faces:
                ids.append(id)
                facesSamples.append(img_numpy[y:y + h, x:x + w])
 
        # 打印脸部特征和id
        print('id:', id)
        print('fs:', facesSamples)
        return facesSamples, ids
 
 
    if __name__ == '__main__':
    # 图片路径
        path = "D:/information/"
    # 获取图像数组和id标签数组和姓名
        faces, ids = getImageAndLabels(path)
    # 获取训练对象
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 训练
        recognizer.train(faces, np.array(ids))
    # 保存文件
        recognizer.write("D:/trainer/trainer2.yml")
def func3():
    recogizer = cv2.face.LBPHFaceRecognizer_create()
    recogizer.read("D:/trainer/trainer2.yml")
    names = []
    warningtime = 0
    def md5(str):
        import hashlib
        m = hashlib.md5()
        m.update(str.encode("utf8"))
        return m.hexdigest()
# 准备识别的图片
    def face_detect_demo(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度
        face_detector = cv2.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
        face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
        # face=face_detector.detectMultiScale(gray)
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
            cv2.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=1)
        # 人脸识别
            ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        # print('标签id:',ids,'置信评分：', confidence)
            if confidence > 80:
                cv2.putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            else:
                cv2.putText(img, str(names[ids - 1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        cv2.imshow('result', img)
    # print('bug:',ids)
    def name():
        path = "D:/information/"
        # names = []
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            name = str(os.path.split(imagePath)[1].split('.', 2)[1])
            names.append(name)
    cap = cv2.VideoCapture(0)
    name()
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        face_detect_demo(frame)
        if ord('3') == cv2.waitKey(1):
            break
    cv2.destroyAllWindows()
    cap.release()
label=Label(root,text="请输入姓名，只可输入英文",font=("楷体",10),fg="red")
#定位
label.grid(row=1,column=0)
entry=Entry(root,font=("宋体",25),fg="blue")
entry.grid(row=2,column=0)
 
#添加点击按钮
button=Button(root,text="信息录入",font=("楷体",25),fg="blue",command=func1)
button.grid(row=3,column=0)
button=Button(root,text="分析数据",font=("楷体",25),fg="blue",command=func2)
button.grid(row=4,column=0)
button=Button(root,text="开始识别",font=("楷体",25),fg="blue",command=func3)
button.grid(row=5,column=0)
# 显示窗口
root.mainloop()