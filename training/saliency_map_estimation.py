import cv2

def saliency_map_estimation(image):

    """
    Wrapper around the opencv-python saliency map methods.
    :return: Saliency map
    """

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    (success, saliencyMap) = saliency.computeSaliency(image)

    return saliencyMap