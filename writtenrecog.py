import imageio
from sklearn import datasets
import numpy
from sklearn.neural_network import MLPClassifier

numpy.set_printoptions(threshold=numpy.nan)

im0 = imageio.imread('0.png')
im1 = imageio.imread('1.png')
im2 = imageio.imread('2.png')
im3 = imageio.imread('3.png')
im4 = imageio.imread('4.png')
im5 = imageio.imread('5.png')
im6 = imageio.imread('6.png')
im7 = imageio.imread('7.png')
im8 = imageio.imread('8.png')
im9 = imageio.imread('9.png')

image0array = im0[:,:,3].astype(float)
image1array = im1[:,:,3].astype(float)
image2array = im2[:,:,3].astype(float)
image3array = im3[:,:,3].astype(float)
image4array = im4[:,:,3].astype(float)
image5array = im5[:,:,3].astype(float)
image6array = im6[:,:,3].astype(float)
image7array = im7[:,:,3].astype(float)
image8array = im8[:,:,3].astype(float)
image9array = im9[:,:,3].astype(float)

image0array = (image0array*16/255)
image1array = (image1array*16/235)
image2array = (image2array*16/255)
image3array = (image3array*16/255)
image4array = (image4array*16/235)
image5array = (image5array*16/255)
image6array = (image6array*16/255)
image7array = (image7array*16/235)
image8array = (image8array*16/255)
image9array = (image9array*16/255)


totalarray = numpy.vstack([
    (image0array,image1array, image2array,image3array,image4array,image5array,image6array,image7array,image8array,image9array)
])

digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

n_samples = len(digits.images)

m_samples = len(totalarray)
data = digits.images.reshape((n_samples, -1))

imagedata = totalarray.reshape((m_samples, -1))

model = MLPClassifier(solver='lbfgs', alpha=0.0001,hidden_layer_sizes=(250, 100), random_state=1)
model.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

expected = digits.target[n_samples // 2:]

predicted = model.predict(imagedata)
expected = [0,1,2,3,4,5,6,7,8,9]
print("expected: "+ str(expected))
print("prediction +" + str(predicted))
accuracy = 0
loop = 0
while (loop <= (m_samples-1)):
    if (int(expected[loop]) == int(predicted[loop])):
        accuracy = accuracy+1

    else:
        accuracy = accuracy

    loop = loop + 1
accuracytotal = (accuracy/m_samples)*100
print("total accuracy: " + str(accuracytotal))
