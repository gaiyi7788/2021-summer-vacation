## Pytorch CTPN
> update 19-03-20 wed: android ocr


This is a pytorch implementation of [CTPN(Detecting Text in Natural Image with Connectionist Text Proposal Network)](https://arxiv.org/pdf/1609.03605.pdf).Inspired by [keras-ocr](https://github.com/xiaomaxiao/keras_ocr).

Training log is available:[Training Log](./logs/training_logs.pdf)(Chinese)

|model|size|
|:--:|:--:|
|keras-CTPN|142M|
|**pytorch-CTPN**|**67.6M**|

### train
- ~~download ctpn model weights (converted from keras ctpn weights) `ctpn_keras_weights.pth.tar` from [dropbox](https://www.dropbox.com/s/81zfc50x6g6fauz/ctpn_keras_weights.pth.tar?dl=0), and move it to **./checkpoints/**~~ (*For a number of reasons, the pretrained weights will no longer be available.Thanks for your attention.*)
- ~~download [VOC2007_text_detection Chinese Text Detection dataset](http://not_available_any_more_due_to_lack_of_space) and move it to **./images/**~~
- run `python ctpn_train.py --image-dir image_dir --labels-dir labels_dir --num-workers num_workers`

### predict
- ~~download the pretrained weights from [dropbox](https://www.dropbox.com/s/r1zjw167a5lsk4l/ctpn_ep18_0.0074_0.0121_0.0195%28w-lstm%29.pth.tar?dl=0)~~
- Please refer to [predict.py](./ctpn_predict.py) for more details.

### results
[Training Log](./logs/training_logs.pdf)(Chinese)

### Android DEMO
These days, I'm working on deploying this model on Android devices.you can check the results from [here](./logs/ANDROID_OCR.pdf).

**Android text recognition 4-23**
> Find out that adopting [skew transform](./results/ANDROID_DETECTION_SKEW.GIF) can significantly improve recognition accuracy.(It may take a few seconds, heavily depends on your harware and input image size)

![reco](./results/ANDROID_RECO_DEMO.GIF)

### reference
- [CTPN (Detecting Text in Natural Image with Connectionist Text Proposal Network)](https://arxiv.org/pdf/1609.03605.pdf)
- [keras-ocr](https://github.com/xiaomaxiao/keras_ocr)

### Licence
[MIT License](https://opensource.org/licenses/MIT)
