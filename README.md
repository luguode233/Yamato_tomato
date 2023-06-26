# Yamato_tomato
内置了实现图像爬取、模型训练和图像分类三个功能模块，将三个功能模块通过tkinter实现交互界面，再通过pyinstaller进行封装
图像爬取选用了百度图片的url，正则表达式爬取
模型训练使用预训练卷积神经网络通过迁移学习进行训练，内置了四种模型供选择
图片分类可以实现调用模型实现对混合图片的分类
使用pyinstaller进行封装，封装命令为
Pyinstaller -F -w -i logo.ico python_tk.py
注意需要创建虚拟环境用于封装，以避免exe过大
