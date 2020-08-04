def expand_image(originial_img, x, y, width, height, beta):
    """
    Function to expand an image by width*(1+beta) and height*(1+beta)
    :param originial_img: Originial image, i.e. not the cropped version.
    :param x,y,width, height: xy coordinates, width and height of the window.
    :param beta: Factor by which to expand the image
    :return: Expanded image.
    """

    # New x and y
    ## 0.5 * beta is added at the left and right of the window.
    x_new = int(max(x - 0.5 * beta * width, 0))
    y_new = int(max(y - 0.5 * beta * height, 0))

    # New height and width are
    width_new = int(width*(1+beta))

    height_new = int(height*(1+beta))

    if x_new + width_new > 4032:
        width_new = 4032 - x_new

    if y_new + height_new > 3024:
        height_new = 3024 - y_new

    expanded_image = originial_img[y_new:y_new+height_new, x_new:x_new+width_new,:]

    return expanded_image

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image = plt.imread("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/test_image.jpg")

    x, y, widht, height = 100, 100, 2000, 1000
    beta = 10

    expanded_image = expand_image(originial_img= image, x = x, y = y, width = widht, height = height, beta = beta)


    x, y, widht, height = 0, 0, 2000, 1000
    beta = -0.1
    expanded_image = expand_image(originial_img=image, x=x, y=y, width=widht, height=height, beta=beta)