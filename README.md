### Flask 后端逻辑

后端接受小程序 upload 附带 formData 数据格式如下:
```python
formData: {
    'mode': 'M',    // 修改上传
    'operator': 0,  // 修改上传
},
```
- mode:str 'L'/'M'/'R' 表示拍照模式，分别代表左侧拍、中间拍、全部拍
- operator:int 0/1 表示操作员侧，分别代表左操作员侧、右操作员侧

回传 json 数据。
```python
result = {
    'area': res_area, # 面积占比
    'region': res_region, # 区域
    'image_path': new_file_path
}
```

- area: list 表示每个灰斑面积占比，需要前端乘系数
- region: list 表示每个灰斑区域分布，和area分别对应
- image_path: str 表示图像存储路径


JS 代码，直接绑定到按钮即可，实现了图像选择和图像上传，数据处理没有用json，直接随着upload一块传上去，需要修改一下传入的值
```javascript
// 文件上传
uploadFile: function () {
    console.log('enter upload file')
    wx.chooseMedia({
        count: 1,
        mediaType: ['image'],
        sourceType: ['album', 'camera'],
        maxDuration: 30,
        camera: 'back',

        success: (res) => {
            if (res.tempFiles.length > 0) {
                const tempFilePath = res.tempFiles[0].tempFilePath;
                wx.uploadFile({
                    filePath: tempFilePath,
                    name: 'file',
                    url: 'http://to-create-future.site:20000/upload',
                    formData: {
                        'mode': 'M',    // 修改上传
                        'operator': 0,  // 修改上传
                    },
                    success: (uploadRes) => {
                        const data = JSON.parse(uploadRes.data);   // data 为服务器回传的结果 参考下方数据
                        console.log('Upload success:', data);
                        this.setData({
                            imageUrl: data.image_path
                        })
                    },
                    fail: (error) => {
                        console.error('Upload failed:', error);
                    }
                })
            } 
            else {
                console.log('No media selected');
            }
        },

        fail: (err) => {
            console.error('Error selecting media:', err);
        }
    })
},
```

### 前端计算base

```txt
50kg/m 全6580  底 2472.75，
60kg/m 全7745  底 2884.39，
60N  全7745    底2884.39 ，
75kg/m  全9503.7   底 3,529.25
```
