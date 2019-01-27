
from Model import Classifier
import numpy as np

data_dimension = 28

X = np.load( 'data{}/x.npy'.format( data_dimension ))
Y = np.load( 'data{}/y.npy'.format( data_dimension ))
test_X = np.load( 'data{}/test_x.npy'.format( data_dimension ))
test_Y = np.load( 'data{}/test_y.npy'.format( data_dimension ))

X = X.reshape( ( X.shape[0] , data_dimension**2  ) ).astype( np.float32 )
test_X = test_X.reshape( ( test_X.shape[0] , data_dimension**2 ) ).astype( np.float32 )

classifier = Classifier( number_of_classes=2 )
classifier.load_model( 'models/model.h5')

parameters = {
    'batch_size' : 250 ,
    'epochs' : 10 ,
    'callbacks' : None ,
    'val_data' : None
}

classifier.fit( X , Y  , hyperparameters=parameters )
classifier.save_model( 'models/0001.h5')

loss , accuracy = classifier.evaluate( test_X , test_Y )
print( "Loss of {}".format( loss ) , "Accuracy of {} %".format( accuracy * 100 ) )
print ( classifier.predict( test_X ).argmax( axis=1 ) )

