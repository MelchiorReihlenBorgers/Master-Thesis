from DataLoad import DataLoad
import matplotlib.pyplot as plt
import numpy as np

def gaussian(xL, yL, H, W, sigma=100):

    channel = [np.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
    channel = np.array(channel, dtype=np.float32)
    channel = np.reshape(channel, newshape=(H, W))

    return channel


if __name__ == "__main__":
    data = DataLoad(path="/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis")
    images, depths, radians = data.load_data(N=1)

    example_heatmap = gaussian(3000, 1400, 3024, 4032, sigma = 100)

    image = plt.imread("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/test_image.jpg")
    plt.imshow(example_heatmap)
    plt.imshow(image, alpha=0.5)
    plt.title("Example Image and Keypoint")
    plt.show()