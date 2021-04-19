# Tensorrt_PSENet

## 1.生成wts权重文件
详见TensorRT_PSENet/attach/psenet2trt_wts/说明.txt  

## 2.配置configure.h
打开configure.h 配置必要的参数  

### 2.1 序列化（SERIALIZE）engine模型
```
#define SERIALIZE    //SERIALIZE   INFER
```
需要配置wts路径和生成engine模型的路径  


### 2.2 推理（INFER）
```
#define INFER    //SERIALIZE   INFER
```
需要配置engine模型路径path_read_engin  
图片文件夹路径dir_img  
b_show 等于1表示看效果  
b_test_save_dir=1表示生成测试精度所需要的txt，需要给生成txt文件夹路径save_path  

## 3.依赖dependence
```
TensorRT7.2.3.4 
OpenCV >= 3.4
libtorch >=1.7.0
cuda10.2
```


## 4.运行步骤
本仓库已经把依赖、模型和测试图片文件夹名字已经上传，下载下来放置一样的目录，可以按照下面步骤跑通完整的demo效果：  
4.1 配置环境  
4.2 生成wts文件（pytorch1.0，py3）  
4.3 序列化（SERIALIZE）生成engine模型  
4.4 推理（INFER）看效果  
4.5 best regards to you  




