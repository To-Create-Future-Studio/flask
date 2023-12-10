### Flask 后端逻辑

后端接受 json 数据格式如下:
```python
data = {
    'image_path': image_path,
    'mode': 'M',
    'operator': 0,
}
```

image_path:str 表示图像绝对路径
mode:str 'L'/'M'/'R' 表示拍照模式，分别代表左侧拍、中间拍、全部拍
operator:int 0/1 表示操作员侧，分别代表左操作员侧、右操作员侧

回传 json 数据。
```python
result = {
    'area': res_area, # 面积占比
    'region': res_region, # 区域
    'image_path': new_file_path
}
```

area: list 表示每个灰斑面积占比，需要前端乘系数
region: list 表示每个灰斑区域分布，和area分别对应
image_path: str 表示图像存储路径
