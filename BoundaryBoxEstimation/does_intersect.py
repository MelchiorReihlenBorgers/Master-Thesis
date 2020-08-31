from training.simulate_windows import simulate_windows


def does_intersect(annotation_x, annotation_y, annotation_width, annotation_height, width_low, height_low):
    """
    Check whether two boxes overlap or not.
    The .csv exported from makesense.ai has the following columns:
        - Label
        - X of top left point
        - Y of top left point
        - width
        - height
        - Image name
        - Xdim image
        - Ydim image

    :param annotation: .csv file obtained from annotating the image on makesense.ai
    :return:    intersection: Bool that indicates whether the two rectangles intersect
                width, height, x, y: Width, height and xy coordinates of top left corner of the simulated rectangle
    """

    width, height, x, y = simulate_windows(N = 1, width_low = width_low, height_low= height_low)

    # Get the top left and bottom right corner of both the annotated window and the simulated window.
    tl_sim = (x, y)
    br_sim = (x + width, y + height)

    tl_annotation = (annotation_x, annotation_y)
    br_annotation = (annotation_x + annotation_width, annotation_y + annotation_height)

    # Code the conditions
    sim_right_of_annotation = tl_sim[0] >= br_annotation[0]
    sim_left_of_annotation = tl_annotation[0] >= br_sim[0]

    ## Be careful here. Images start at zero, zero in the top left corner -->  going to the bottom left corner means going up the y values.
    sim_above_annotation = br_sim[1] <= tl_annotation[1]
    sim_below_annotation = br_annotation[1] <= tl_sim[1]

    # If one rectangle is on left side of other
    if (sim_right_of_annotation or sim_left_of_annotation):
        intersection = False

    # If one rectangle is above other
    if (sim_above_annotation or sim_below_annotation):
        intersection = False


    intersection = True

    return intersection, width, height, x, y

if __name__ == "__main__":
    # Do both rectangles intersect?
    print(does_intersect(annotation_x=1, annotation_y=10, annotation_height=10, annotation_width=10)[ 0 ])