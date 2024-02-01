## Detection and quantitative analysis of patient-ventilator interactions in ventilated infants by convolutional neural networks

Authors : David Chong (chongdtwdavid94@gmail.com), Gusztav Belteki (gbelteki@icloud.com)

This is the accompanying code repository for the titular publication. The trained models can be found in the model_dev/model_checkpoints folder.

To load and use the models you can use the example pipeline under [model_dev/asynchrony_classification_pipeline.py](https://github.com/chongtwd/Detection-and-quantitative-analysis-of-patient-ventilator-interactions-in-ventilated-neonates/blob/main/model_dev/asynchrony_classification_pipeline.py) as a starting point. The training and checkpoints depend on the following libraries in addition to other ones which you most likely already have installed.

1. torch
2. pytorch_lightning
3. torchmetrics

The models were trained using [Ventiliser](https://github.com/barrinalo/ventiliser) as the segmenting algorithm, so you may wish to use that to preprocess the waveforms.
