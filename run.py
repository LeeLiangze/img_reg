from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()
prediction = ImagePrediction()

# fast prediction time and high accuracy
# prediction.setModelTypeAsResNet()
# prediction.setModelPath(os.path.join(execution_path, "model/resnet50_weights_tf_dim_ordering_tf_kernels.h5"))

# slow prediction time and higher accuracy
prediction.setModelTypeAsDenseNet()
prediction.setModelPath(os.path.join(execution_path, "model/DenseNet-BC-121-32.h5"))

# slower prediction time and highest accuracy
# prediction.setModelTypeAsInceptionV3()
# prediction.setModelPath(os.path.join(execution_path, "model/inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))

prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "image/knife6_easy.PNG"), result_count=10 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
