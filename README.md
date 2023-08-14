# textlab
This repo features natural language processing pipeline for dialog engines

# Tips
Batch size affects training speed

Trial 1 summary
Hyperparameters:
units: 352
learning_rate: 0.00041615784427578373
Score: 0.3911738197008769

Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, None)]            0         
                                                                 
 embedding_1 (Embedding)     (None, None, 32)          4800000   
                                                                 
 spatial_dropout1d_1 (Spatia  (None, None, 32)         0         
 lDropout1D)                                                     
                                                                 
 bidirectional_2 (Bidirectio  (None, None, 704)        1084160   
 nal)                                                            
                                                                 
 bidirectional_3 (Bidirectio  (None, 704)              2976512   
 nal)                                                            
                                                                 
 dense_1 (Dense)             (None, 17)                11985     
                                                                 
=================================================================
Total params: 8,872,657
Trainable params: 8,872,657
Non-trainable params: 0
_________________________________________________________________