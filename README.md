# captcha-solving-ctc
comparision of CTC and native output encoding + flask service

implementation of https://keras.io/examples/vision/captcha_ocr/

the techiques used:
- custom loss (layout approach)
- tensorflow conditional execution
- tensorflow batch operator
- python subprocess pipe communication

```
Number of images found:  18853
Number of labels found:  18853
Number of unique characters:  20
Characters present:  ['2', '4', '5', '6', '7', '8', '9', 'б', 'в', 'г', 'д', 'ж', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т']
(16967,)
```

# CTC encoding
Dense (50, 20) -> 

Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25)) - receive only (n1,n2) shape, whre n1 - called "input sequence"/"timestamps"

->

CTCLayer - do not modify y_pred from dense2


comparision of LSTM
- lstm (LSTM)                    (None, 50, 64)       24832       ['dropout_1[0][0]'] - return_sequences=True
- lstm (LSTM)                    (None, 64)           24832       ['dropout_1[0][0]'] - output corresponding to the last timestep, containing information about the entire input sequence
- bidirectional (Bidirectional)  (None, 50, 128)      49664       ['dropout_1[0][0]']   - bidirectual

CTC uses keras.backend.ctc_decode to decode softmax output trained with keras.backend.ctc_batch_cost loss.
```
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 image (InputLayer)             [(None, 200, 60, 1)  0           []                               
                                ]                                                                 
                                                                                                  
 Conv1 (Conv2D)                 (None, 200, 60, 16)  160         ['image[0][0]']                  
                                                                                                  
 pool1 (MaxPooling2D)           (None, 100, 30, 16)  0           ['Conv1[0][0]']                  
                                                                                                  
 Conv2 (Conv2D)                 (None, 100, 30, 32)  4640        ['pool1[0][0]']                  
                                                                                                  
 dropout (Dropout)              (None, 100, 30, 32)  0           ['Conv2[0][0]']                  
                                                                                                  
 pool2 (MaxPooling2D)           (None, 50, 15, 32)   0           ['dropout[0][0]']                
                                                                                                  
 reshape (Reshape)              (None, 50, 480)      0           ['pool2[0][0]']                  
                                                                                                  
 dense1 (Dense)                 (None, 50, 20)       9620        ['reshape[0][0]']                
                                                                                                  
 dropout_1 (Dropout)            (None, 50, 20)       0           ['dense1[0][0]']                 
                                                                                                  
 bidirectional (Bidirectional)  (None, 50, 128)      43520       ['dropout_1[0][0]']              
                                                                                                  
 label (InputLayer)             [(None, None)]       0           []                               
                                                                                                  
 dense2 (Dense)                 (None, 50, 22)       2838        ['bidirectional[0][0]']          
                                                                                                  
 ctc_loss (CTCLayer)            (None, 50, 22)       0           ['label[0][0]',                  
                                                                  'dense2[0][0]']                 
                                                                                                  
==================================================================================================
Total params: 60,778
Trainable params: 60,778
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/20

16967/16967 [==============================] - 158s 9ms/step - loss: 7.4851 - val_loss: 1.8812
Epoch 2/20
16967/16967 [==============================] - 130s 8ms/step - loss: 2.6040 - val_loss: 1.4825
Epoch 3/20
16967/16967 [==============================] - 128s 8ms/step - loss: 2.0989 - val_loss: 1.1787
Epoch 4/20
16967/16967 [==============================] - 128s 8ms/step - loss: 1.8145 - val_loss: 1.0489
Epoch 5/20
16967/16967 [==============================] - 129s 8ms/step - loss: 1.6860 - val_loss: 1.0097
Epoch 6/20
16967/16967 [==============================] - 129s 8ms/step - loss: 1.5893 - val_loss: 0.9703
Epoch 7/20
16967/16967 [==============================] - 130s 8ms/step - loss: 1.5364 - val_loss: 0.9351
Epoch 8/20
16967/16967 [==============================] - 129s 8ms/step - loss: 1.4614 - val_loss: 0.8324
Epoch 9/20
16967/16967 [==============================] - 130s 8ms/step - loss: 1.4057 - val_loss: 0.9483

Positive 1047
test examples 1159
Accuracy 0.903364969801553
```

# Naive categorical encoding

```
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 image (InputLayer)             [(None, 200, 60, 1)  0           []                               
                                ]                                                                 
                                                                                                  
 Conv1 (Conv2D)                 (None, 200, 60, 16)  160         ['image[0][0]']                  
                                                                                                  
 pool1 (MaxPooling2D)           (None, 100, 30, 16)  0           ['Conv1[0][0]']                  
                                                                                                  
 Conv2 (Conv2D)                 (None, 100, 30, 32)  4640        ['pool1[0][0]']                  
                                                                                                  
 dropout (Dropout)              (None, 100, 30, 32)  0           ['Conv2[0][0]']                  
                                                                                                  
 pool2 (MaxPooling2D)           (None, 50, 15, 32)   0           ['dropout[0][0]']                
                                                                                                  
 reshape (Reshape)              (None, 50, 480)      0           ['pool2[0][0]']                  
                                                                                                  
 dense1 (Dense)                 (None, 50, 32)       15392       ['reshape[0][0]']                
                                                                                                  
 dropout_1 (Dropout)            (None, 50, 32)       0           ['dense1[0][0]']                 
                                                                                                  
 flatten (Flatten)              (None, 1600)         0           ['dropout_1[0][0]']              
                                                                                                  
 dense11 (Dense)                (None, 120)          192120      ['flatten[0][0]']                
                                                                                                  
 reshape2 (Reshape)             (None, 6, 20)        0           ['dense11[0][0]']                
                                                                                                  
 dense12 (Dense)                (None, 2)            3202        ['flatten[0][0]']                
                                                                                                  
 label (InputLayer)             [(None, 6, 20)]      0           []                               
                                                                                                  
 count (InputLayer)             [(None, 2)]          0           []                               
                                                                                                  
 loss_layer (LossLayer)         (None, 6, 20)        0           ['reshape2[0][0]',               
                                                                  'dense12[0][0]',                
                                                                  'label[0][0]',                  
                                                                  'count[0][0]']                  
                                                                                                  
==================================================================================================
Total params: 215,514
Trainable params: 215,514
Non-trainable params: 0
__________________________________________________________________________________________________

16965/16967 [============================>.] - ETA: 0s - loss: 0.3450Tensor("cond/Identity:0", shape=(6, 20), dtype=float32)
Tensor("cond_1/Identity:0", shape=(6, 20), dtype=float32)
16967/16967 [==============================] - 83s 5ms/step - loss: 0.3450 - val_loss: 0.2518
Epoch 2/100
16967/16967 [==============================] - 79s 5ms/step - loss: 0.2362 - val_loss: 0.2172
Epoch 3/100
16967/16967 [==============================] - 94s 6ms/step - loss: 0.2073 - val_loss: 0.2149
Epoch 4/100
16967/16967 [==============================] - 95s 6ms/step - loss: 0.1930 - val_loss: 0.2045
Epoch 5/100
16967/16967 [==============================] - 97s 6ms/step - loss: 0.1837 - val_loss: 0.1997
Epoch 6/100/46
16967/16967 [==============================] - 104s 6ms/step - loss: 0.1760 - val_loss: 0.1960
Epoch 7/100
16967/16967 [==============================] - 93s 5ms/step - loss: 0.1725 - val_loss: 0.1986

Positive 128
test examples 1159
Accuracy 0.11044003451251079

```

