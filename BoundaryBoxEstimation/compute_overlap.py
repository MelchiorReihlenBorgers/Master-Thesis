from training.compute_area import  compute_area

def compute_overlap(x1, y1, width1, height1, x2, y2, width2, height2):
    """
    Get the top left coordinate of the overlap rectangle, the width and the height.

    :param x1, y1, width1, height1: xy coordinates, width and height of first rectangle
    :param x2, y2, width2, height2: xy coordinates, width and height of second rectangle

    :return: Size of the overlapping area
    """

    A1 = (x1, y1)
    A2 = (x1 + width1, y1 + height1)

    B1 = (x2, y2)
    B2 = (x2 + width2, y2 + height2)

    X = min(A2[0], B2[0]) - max(A1[0], B1[0])
    Y = min(A2[1], B2[1]) - max(A1[1], B1[1])

    overlapping_area = compute_area(X, Y)

    return overlapping_area

if __name__ == "__main__":
    x1, y1, width1, height1 = 10, 10, 2, 20
    x2, y2, width2, height2 = 11, 20, 2, 20


    overlap = compute_overlap(x1, y1, width1, height1, x2, y2, width2, height2)

    print("The overlap is: {}".format(overlap))
