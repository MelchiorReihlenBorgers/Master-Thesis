import os
from DataLoad import DataLoad
import matplotlib.pyplot as plt
import cv2

# Load the data
path = os.getcwd()
data = DataLoad(path=os.getcwd())

n_total = len(data.extract_paths()[0])  # If you want to load all images.

images, depths, radians = data.load_data(N=3)


saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
saliency2 = cv2.saliency.StaticSaliencyFineGrained_create()

(success, saliencyMap) = saliency.computeSaliency(images[0])
(success2, saliencyMap2) = saliency2.computeSaliency(images[0])

saliencyMap = (saliencyMap2 * 255).astype("uint8")

threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


plt.imshow(threshMap)