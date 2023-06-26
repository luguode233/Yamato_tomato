#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import filedialog, messagebox,ttk
from PIL import ImageTk, Image
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
import threading
import re
import requests
import traceback
import time
import json
import shutil
def switch_windows():
    # 隐藏第一个窗口
    root.withdraw()
    # 显示第二个窗口
    window.deiconify()
def switch_windows2():
    # 隐藏第一个窗口
    window.withdraw()
    # 显示第二个窗口
    root.deiconify()
def switch_windows3():
    # 隐藏第一个窗口
    root.withdraw()
    # 显示第三个窗口
    window_root.deiconify()
def switch_windows4():
    # 隐藏第一个窗口
    window_root.withdraw()
    # 显示第二个窗口
    root.deiconify()
def open_folder():
    global selected_folder_path, data, labels
    # 选择文件夹
    selected_folder_path = filedialog.askdirectory()
    # 清空数据
    data = []
    labels = []
    # 显示文件夹路径
    folder_text.delete(1.0, tk.END)
    folder_text.insert(tk.END, selected_folder_path)
def process_folder():
    global selected_folder_path, data, labels, train_data, train_labels, class_names,label_to_class_dict
    if selected_folder_path == "":
        messagebox.showerror("错误", "请先选择文件夹！")
        return
    class_names = sorted(os.listdir(selected_folder_path))
    for i, class_name in enumerate(class_names):
        label_to_class_dict[i] = class_name
        class_dir = os.path.join(selected_folder_path, class_name)
        j = 0
        # 遍历文件夹中的所有jpg文件，然后转化为224x224
        for img_name in os.listdir(class_dir):
            j += 1
            img_path = os.path.join(class_dir, img_name)
            img = image.load_img(img_path, target_size=(img_size, img_size))
            x = image.img_to_array(img)
            data.append(x)
            labels.append(i)
            if j == 500:
                break
    # 将列表转换为numpy数组
    train_data = np.array(data)
    train_labels = np.array(labels)
    messagebox.showinfo("提示", "读取"+selected_folder_path+" 完成")
def train_model_thread():
    global max_epoch
    max_epoch = 0
    progress_model["value"] = max_epoch
    threading.Thread(target=train_model).start()
def accuracy_callback(epoch, logs):
    global max_epoch
    max_epoch += 1
    progress_model["value"] = max_epoch
    progress_model.update()
    time.sleep(0.1)
def train_model():
    global train_labels, history,save_folder,model_get
    
    save_folder = "models"
    os.makedirs(save_folder, exist_ok=True)
    train_labels = to_categorical(train_labels, len(class_names))
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=test_size_change)
    if model_get == "ResNet50":
            # 加载ResNet50模型
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    elif model_get == "DenseNet121":
            # 加载DenseNet121模型
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    elif model_get == "InceptionV3":
            # 加载INceptionV3模型
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    elif model_get == "MobileNet":
        # 加载MobileNet模型:
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    freeze_layers = 60
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation='relu')(x)
    predictions = Dense(len(class_names), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # 编译模型
    model.compile(optimizer=optimizers.RMSprop(learning_rate=learning_rate_change), loss='categorical_crossentropy', metrics=['accuracy'])
    model.class_names = class_names
    # 定义回调函数
    early_stop = EarlyStopping(monitor='val_accuracy', patience=6, mode='max')
    model_checkpoint = ModelCheckpoint(os.path.join(save_folder, "best_model.h5"), monitor='val_accuracy', save_best_only=True, mode='max')
    # 训练模型
    batch_size = fenpi_change
    epochs = diedai_change
    progress_model["maximum"] = epochs 
    accuracy_lambda_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=accuracy_callback)
    train_datagen = ImageDataGenerator(rotation_range=20)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[early_stop, model_checkpoint,accuracy_lambda_callback],verbose=0)
    dict_file_path = os.path.join(save_folder, "label_to_class_dict.json")
    with open(dict_file_path, "w") as file:
        json.dump(label_to_class_dict, file)
    messagebox.showinfo("训练完成", "模型训练完成！")
    history_text.delete(1.0, tk.END)
    history_text.insert(tk.END, "训练历史信息:\n")
    for epoch, acc, loss, val_acc, val_loss in zip(range(len(history.history['accuracy'])),
                                                   history.history['accuracy'],
                                                   history.history['loss'],
                                                   history.history['val_accuracy'],
                                                   history.history['val_loss']):
        history_text.insert(tk.END, f"Epoch {epoch+1}/{epochs}\n")
        history_text.insert(tk.END, f"训练集准确率: {acc:.4f}\n")
        history_text.insert(tk.END, f"训练集损失: {loss:.4f}\n")
        history_text.insert(tk.END, f"验证集准确率: {val_acc:.4f}\n")
        history_text.insert(tk.END, f"验证集损失: {val_loss:.4f}\n\n")
    progress_model["value"] = epochs
def clicked():
    global test_size_change,learning_rate_change,diedai_change,fenpi_change
    if selected.get() ==  '':
        selected_moren = '7:3'
        test_size_change = 0.3
    elif selected.get() == '7:3':
        test_size_change = 0.3
    elif selected.get() == '8:2':
        test_size_change = 0.2
    elif selected.get() == '6:4':
        test_size_change = 0.4
    learning_rate_change = float(selected2.get())
    diedai_change  = int(cishu.get())
    fenpi_change = int(cishu2.get())
    model_get = combo.get()
    lbl.configure(text=("目前设置训练集与测试集比例为"+str(selected.get())+'选用预训练神经网络为'+str(model_get)+'\n学习率为'+str(selected2.get())+"，迭代次数为"+str(cishu.get())+"，训练批次为"+str(cishu2.get())))
def open_folder2():
    global path_pacon
    # 选择文件夹
    path_pacon = filedialog.askdirectory()
    # 显示文件夹路径
    folder_text2.delete(1.0, tk.END)
    folder_text2.insert(tk.END, path_pacon)    
def start_crawling():
    global total_pages,Pq_Number
    Pq_Number = int(var.get())
    total_pages = Pq_Number // 60 + 2
    progress["maximum"] = Pq_Number

    # 创建一个新线程来执行爬取操作
    t = threading.Thread(target=crawl_images).start()

def crawl_images():
    global words,word,root2
    words = []
    progress["value"] = 0
    headers = {'user-agent': 'Mozilla/5.0'}
    words = [txt.get()]
    root2 = path_pacon
    if not os.path.exists(root2):
        os.mkdir(root2)
    for word in words:
        lastNum = 0
        pageId = 0
        for page in range(total_pages):
            url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + "&pn=" + str(
                pageId) + "&gsm=?&ct=&ic=0&lm=-1&width=0&height=0"
            pageId += 60
            html = requests.get(url, headers=headers)
            lastNum = dowmloadPic(html.text, word, lastNum)
    # 爬取完成后重置进度条的值
    progress["value"] = Pq_Number

def dowmloadPic(html, keyword, startNum):
    global path, img, f, um, each, pic_url
    headers = {'user-agent': 'Mozilla/5.0'}
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)
    num = len(pic_url)
    i = startNum
    subroot = root2 + '/' + word

    print('找到关键词:' + word + '的图片，现在开始下载图片...')

    for each in pic_url:
        a = '第' + str(i + 1) + '张图片，图片地址:' + str(each) + '\n'
        b = '正在下载' + a
        print(b)
        path = subroot + '/' + str(i + 1)
        print(path)
        try:
            if not os.path.exists(subroot):
                os.mkdir(subroot)
            if not os.path.exists(path):
                pic = requests.get(each, headers=headers, timeout=10)
                with open(path + '.jpg', 'wb') as f:
                    f.write(pic.content)
                    f.close()
        except:
            traceback.print_exc()
            print('【错误】当前图片无法下载')
            continue
        try:
            img = Image.open(path + '.jpg')
        except:
            os.remove(path + '.jpg')
            continue
        progress["value"] = i + 1
        progress.update()
        time.sleep(0.2)  # 模拟爬取一页数据的延迟
        if i >= Pq_Number - 1:
            break
        i += 1

    return i
def open_tupian():
    global filepath,image_files,img,photo
    filepath = filedialog.askopenfilename()
    if filepath:
        try:
            img = Image.open(filepath)
        except:
            messagebox.showwarning("警告", "选择的文件不是有效的图片文件！")
            return
    img = image.load_img(filepath, target_size=(img_size, img_size))
def open_folder4():
    global img_path_all,aaaa,data,path_hunhe
    # 选择文件夹
    path_hunhe = filedialog.askdirectory()
    # 清空数据
    aaaa=[]
    data = []
    for img_name in os.listdir(path_hunhe):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(path_hunhe, img_name)
            img_path_all.append(img_path)
            img = image.load_img(img_path, target_size=(img_size, img_size))
            x = image.img_to_array(img)
            data.append(x)
    aaaa = np.array(data)
    progress_fenlei["maximum"] = len(aaaa)
    # 显示文件夹路径
    folder_text4.delete(1.0, tk.END)
    folder_text4.insert(tk.END, path_hunhe)   
def open_folder3():
    global save_foldera,model,digit_to_type_dict
    # 选择文件夹
    save_folder = filedialog.askdirectory()
    # 显示文件夹路径
    model = tf.keras.models.load_model(os.path.join(save_folder, "best_model.h5"))
    dict_file_path = os.path.join(save_folder, "label_to_class_dict.json")
    with open(dict_file_path, "r") as file:
        digit_to_type_dict = json.load(file)
    folder_text3.delete(1.0, tk.END)
    folder_text3.insert(tk.END, save_folder)    
def model_doi():
    global aaa,aaaa
    aaa=[]
    img_size=224
    filename=filepath
    img = image.load_img(filename, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    aaa.append(x)
    aaaa = np.array(aaa)
    pred=model.predict(aaaa,verbose=0) #将测试图像送入模型，进行预测
    end = digit_to_type_dict.get(str(pred.argmax()))
    messagebox.showwarning("结果", end)
def model_doi_all():
    global aaaa,pred
    pred = model.predict(aaaa,verbose=0)
    for i in range(len(aaaa)):
        jieguo = pred[i].argmax()
        label = digit_to_type_dict.get(str(jieguo))
        dest_folder = os.path.join(path_hunhe, label)  # 创建目标文件夹路径
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)  # 创建目标文件夹
        src_file = img_path_all[i]  # 使用之前保存的图像路径
        dest_file = os.path.join(dest_folder, os.path.basename(src_file))
        shutil.copy(src_file, dest_file)  # 复制图片到目标文件夹
        progress_fenlei["value"] = i + 1
        progress_fenlei.update()
        time.sleep(0.05)
    progress_fenlei["value"] =len(data)
    messagebox.showwarning("提示", "图片集分类完成！")
#设置变量
data = []
labels = []
label_to_class_dict = {}
img_size = 224
selected_folder_path = ""
train_data = None
train_labels = None
class_names = None
history = None
test_size_change = 0.3
learning_rate_change = 0.001
diedai_change = 30
fenpi_change = 6
model_get = "ResNet50"
img_path_all = []
# 创建第一个窗口
root = tk.Tk()
root.title('模型训练')
root.geometry('1000x800')
# 创建选择文件夹的按钮
open_button = tk.Button(root, text='选择图片文件夹', width=12, command=open_folder)
open_button.place(relx=0.9, rely=0.1, anchor='center')
folder_label = tk.Label(root, text="选择的文件夹路径：")
folder_label.place(relx=0.06, rely=0.1, anchor='center')
# 创建文本框用于显示选择的文件夹路径
folder_text = tk.Text(root, height=1, width=90)
folder_text.place(relx=0.45, rely=0.1, anchor='center')
# 创建处理文件夹的按钮
process_button = tk.Button(root, text='读取图片文件夹', width=12, command=process_folder)
process_button.place(relx=0.9, rely=0.2, anchor='center')


lbl = tk.Label(root, text="目前设置是默认训练集与测试集比例为7：3，选用预训练神经网络为ResNet50，\n迭代次数为30，训练批次设置为6，学习率为0.001",height=40)
lbl.place(relx=0.25, rely=0.6, anchor='center')
train_button = tk.Label(root, text='训练集与测试集比例', width=18)
train_button.place(relx=0.2, rely=0.3, anchor='center')
selected = tk.StringVar()    #保存选项结果
selected.set('7:3')
diedai_73 = tk.Radiobutton(root, text="7:3", value="7:3", variable=selected)
diedai_64 = tk.Radiobutton(root, text="6:4", value="6:4", variable=selected)
diedai_82 = tk.Radiobutton(root, text="8:2", value="8:2", variable=selected)
diedai_73.place(relx=0.1, rely=0.35, anchor='center')
diedai_64.place(relx=0.2, rely=0.35, anchor='center')
diedai_82.place(relx=0.3, rely=0.35, anchor='center')
train_button2 = tk.Label(root, text='学习率', width=18)
train_button2.place(relx=0.2, rely=0.4, anchor='center')
selected2 = tk.StringVar()
selected2.set(0.001)
xuexilv_001 = tk.Radiobutton(root, text="0.001", value=0.001, variable=selected2)
xuexilv_0001 = tk.Radiobutton(root, text="0.0004", value=0.0004, variable=selected2)
xuexilv_00001 = tk.Radiobutton(root, text="0.0001", value=0.0001, variable=selected2)
xuexilv_001.place(relx=0.1, rely=0.45, anchor='center')
xuexilv_0001.place(relx=0.2, rely=0.45, anchor='center')
xuexilv_00001.place(relx=0.3, rely=0.45, anchor='center')
train_button = tk.Label(root, text='迭代次数', width=18)
train_button.place(relx=0.13, rely=0.5, anchor='center')
cishu = tk.StringVar()
cishu.set(30)
spin = tk.Spinbox(root, from_=1,to=105800,width=5, textvariable=cishu)
spin.place(relx=0.13, rely=0.55, anchor='center')
train_button2 = tk.Label(root, text='训练批次', width=18)
train_button2.place(relx=0.25, rely=0.5, anchor='center')
cishu2 = tk.StringVar()
cishu2.set(6)
spin2 = tk.Spinbox(root, from_=1, to=100, width=5, textvariable=cishu2)
spin2.place(relx=0.25, rely=0.55, anchor='center')
train_button = tk.Label(root, text='预训练卷积神经网络选择', width=18)
train_button.place(relx=0.1, rely=0.25, anchor='center')
combo = ttk.Combobox(root)
combo['values'] = ("ResNet50","DenseNet121","InceptionV3","MobileNet")
combo.current(0)
combo.place(relx=0.28, rely=0.25, anchor='center')
tiaozheng_button = tk.Label(root, text='参数设置', width=12,font = 20)
tiaozheng_button.place(relx=0.2, rely=0.2, anchor='center')
btn = tk.Button(root, text="参数保存", width=12,command=clicked)    #保存选项
btn.place(relx=0.2, rely=0.7, anchor='center')
# 创建训练模型的按钮
train_button = tk.Button(root, text='开始训练', width=12, command=train_model_thread)
train_button.place(relx=0.6, rely=0.8, anchor='center')
history_text_button = tk.Label(root, text='训练结果', width=18)
history_text_button.place(relx=0.6, rely=0.22, anchor='center')
history_text = tk.Text(root, height=30, width=35)
history_text.place(relx=0.6, rely=0.5, anchor='center')
progress_model = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate")
progress_model.place(relx=0.4,rely=0.8, anchor='center')


# 创建一个按钮，点击按钮时触发跳转
btn_switch = tk.Button(root, text="切换爬虫界面", command=switch_windows, width=20)
btn_switch.place(relx=0.25,rely=0.85, anchor='center')
btn_switch = tk.Button(root, text="图像分类界面", command=switch_windows3,width=20)
btn_switch.place(relx=0.85, rely=0.85, anchor='center')


window = tk.Toplevel()
window.title("图片爬虫")
window.geometry("600x400")
window.withdraw()
btn_switch2 = tk.Button(window, text="返回", command=switch_windows2)
btn_switch2.place(relx=0.8, rely=0.75, anchor='center')


open_button2 = tk.Button(window, text='选择储存文件夹', width=16, command=open_folder2)
open_button2.place(relx=0.84, rely=0.1, anchor='center')
folder_label2 = tk.Label(window, text="选择的文件夹路径：")
folder_label2.place(relx=0.1, rely=0.08, anchor='center')
folder_text2 = tk.Text(window, height=1, width=30)
folder_text2.place(relx=0.45, rely=0.1, anchor='center')

lbl_pachon = tk.Label(window, text="想要爬取的图片")
lbl_pachon.place(relx=0.1, rely=0.35, anchor='center')
txt = tk.Entry(window, width=10)
txt.place(relx=0.3, rely=0.35, anchor='center')

folder_label2 = tk.Label(window, text="爬取数量")
folder_label2.place(relx=0.1, rely=0.42, anchor='center')
var = tk.StringVar()
var.set("50")
spin = tk.Spinbox(window, from_=0, to=10000, width=5, textvariable=var)
spin.place(relx=0.3, rely=0.42, anchor='center')

folder_label2 = tk.Label(window, text="爬取进度")
folder_label2.place(relx=0.5,rely=0.4, anchor='center')
progress = ttk.Progressbar(window, orient="horizontal", length=200, mode="determinate")
progress.place(relx=0.5,rely=0.55, anchor='center')

start_button = tk.Button(window, text="开始爬取", command=start_crawling)
start_button.place(relx=0.5,rely=0.8, anchor='center')

window_root = tk.Toplevel()
window_root.title("图像分类")
window_root.geometry("800x600")
window_root.withdraw()

open_button = tk.Button(window_root, text='选择模型所在文件夹', width=18, command=open_folder3)
open_button.place(relx=0.88, rely=0.1, anchor='center')
folder_label = tk.Label(window_root, text="选择的文件夹路径：")
folder_label.place(relx=0.1, rely=0.1, anchor='center')
# 创建文本框用于显示选择的文件夹路径
folder_text3 = tk.Text(window_root, height=1, width=60)
folder_text3.place(relx=0.5, rely=0.1, anchor='center')

open_button = tk.Button(window_root, text='选择混合图片集', width=18, command=open_folder4)
open_button.place(relx=0.9, rely=0.3, anchor='center')
folder_label = tk.Label(window_root, text="混合图片集所在路径")
folder_label.place(relx=0.6, rely=0.35, anchor='center')
# 创建文本框用于显示选择的文件夹路径
folder_text4 = tk.Text(window_root, height=1, width=45)
folder_text4.place(relx=0.6, rely=0.3, anchor='center')

button = tk.Button(window_root, text='选择单张图片', width=10, command=open_tupian)
button.place(relx=0.2, rely=0.3, anchor='center')
button = tk.Button(window_root, text='开始分类', width=10, command=model_doi)
button.place(relx=0.2, rely=0.5, anchor='center')
button = tk.Button(window_root, text='开始图片集分类', width=18, command=model_doi_all)
button.place(relx=0.85, rely=0.5, anchor='center')

progress_fenlei = ttk.Progressbar(window_root, orient="horizontal", length=200, mode="determinate")
progress_fenlei.place(relx=0.6,rely=0.5, anchor='center')

btn_switch = tk.Button(window_root, text="返回", command=switch_windows4)
btn_switch.place(relx=0.9, rely=0.85, anchor='center')


root.mainloop()


# In[ ]:




