'''
A program to compute exactly what fraction of an
array of pixels falls inside a circular aperture.

Call as:
[all_x, all_y, all_a] = circ_app.circular_aperture(arrs, circle_loc, rcirc,
                        normalize=True, plot=False, plot_marker_size=5000)

args:
- arrs: Tuple of x and y indexes (from np.meshgrid)
- circle_loc: Tuple containing the x and y coordinate of the
centre of the circle (in pixel units)
- rcirc: The radius fo the circle (in pixel units)
kwargs:
- normalize - if true, the fraction of each pixel inside or outside the
circle is returned. Otherwise, the area is returned. Default is True.
 - plot - if True an image is produced which shows the fractions of each pixel
 inside the aperture. Default is False.
 - plot_marker_size - determines the marker size used in the optional plot.
output:
- all_x: The x-coordinates of the pixels in a 1-D list
- all_y: The y-coordinates of the pixels in a 1-D list
- all_a: The area or fraction of each pixel falling inside the aperture

Writen by Catherine de Burgh-Day. Contact: catherine.dbd@gmail.com

'''

# The below command disables any errors pertaining to missing module attributes.
# This is here becuase pylint has a bug where it can't find module attributes properly!
# pylint: disable=E
# Delete this line (not just comment - it works while commented) to see all errors with pylint

import numpy as np
import matplotlib.pyplot as plt
import os

__author__ = 'Catherine de Burgh-Day'


def rescale_array(arr):
    """
    Rescale or normalise an array to be between 0 and 1

    :param arr:
    :return:
    """
    arr = np.array(arr)
    arr_shape = arr.shape
    arr = arr.flatten()
    min_arr_val = min(arr)
    max_arr_val = max(arr)
    scaled_arr = np.zeros_like(arr)
    for i, elem in enumerate(arr):
        scaled_arr[i] = (elem - min_arr_val) / (max_arr_val - min_arr_val)
    scaled_arr = np.reshape(scaled_arr, arr_shape)
    return scaled_arr


def intersect(vert_a, vert_b, circle_rad):
    """
    Identifies the intersects
    between a circle and a square
    (based on http://mathworld.wolfram.com/Circle-LineIntersection.html)

    :param vert_a:
    :param vert_b:
    :param circle_rad:
    :return:
    """
    vert_a_x, vert_a_y = vert_a
    vert_b_x, vert_b_y = vert_b
    delta_x = vert_b_x - vert_a_x
    delta_y = vert_b_y - vert_a_y
    delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2)
    discriminant = vert_a_x * vert_b_y - vert_b_x * vert_a_y

    # Do this because np.sign(0.0) = 0.0 (not 1.0 or -1.0)
    sdy = delta_y
    if delta_y == 0.0:
        sdy = 1.
    sign_dy = np.sign(sdy)

    intx_1 = (1 / delta_r ** 2)\
             * (discriminant * delta_y - sign_dy * delta_x
                * np.sqrt((circle_rad ** 2)
                          * (delta_r ** 2) - (discriminant ** 2)))
    intx_2 = (1 / delta_r ** 2)\
             * (discriminant * delta_y + sign_dy * delta_x
                * np.sqrt((circle_rad ** 2)
                          * (delta_r ** 2) - (discriminant ** 2)))

    inty_1 = (1 / delta_r ** 2) * (-discriminant * delta_x + abs(delta_y)
                              * np.sqrt((circle_rad ** 2) * (delta_r ** 2)
                                        - (discriminant ** 2)))
    inty_2 = (1 / delta_r ** 2) * (-discriminant * delta_x - abs(delta_y)
                              * np.sqrt((circle_rad ** 2) * (delta_r ** 2)
                                        - (discriminant ** 2)))

    delta = (circle_rad ** 2) * (delta_r ** 2) - (discriminant ** 2)

    if delta > 0:
        is_int = True
    else:
        is_int = False

    if intx_1 < intx_2:
        intx_pos = intx_2
        intx_neg = intx_1
    else:
        intx_pos = intx_1
        intx_neg = intx_2
    if inty_1 < inty_2:
        inty_pos = inty_2
        inty_neg = inty_1
    else:
        inty_pos = inty_1
        inty_neg = inty_2

    return intx_pos, intx_neg, inty_pos, inty_neg, is_int


def circular_seg(int_a, int_b, circle_rad):
    """
    Computes the area of a circular segment

    :param int_a:
    :param int_b:
    :param circle_rad:
    :return:
    """
    int_a_x, int_a_y = int_a
    int_b_x, int_b_y = int_b
    distance = np.sqrt((int_a_x - int_b_x) ** 2 + (int_a_y - int_b_y) ** 2)
    angle = 2.0 * np.arcsin(0.5 * distance / circle_rad)
    a_circ_seg = abs(0.5 * (circle_rad ** 2) * (angle - np.sin(angle)))
    return a_circ_seg


def triangle(vert_a, vert_b, vert_c):
    """
    Computes the area of a triangle

    :param vert_a:
    :param vert_b:
    :param vert_c:
    :return:
    """
    vert_a_x, vert_a_y = vert_a
    vert_b_x, vert_b_y = vert_b
    vert_c_x, vert_c_y = vert_c
    a_tri = abs((vert_a_x * (vert_b_y - vert_c_y) +
                 vert_b_x * (vert_c_y - vert_a_y) +
                 vert_c_x * (vert_a_y - vert_b_y)) / 2.0)
    # area = (1/2)b*h
    #      /\A
    #     /  \
    #    /    \
    #  B/______\C

    return a_tri


def rectangle(vert_a, vert_d):
    """
    Computes the area of a rectangle
    vert_A and vert_D must be opposite i.e. 1 and 3 or 2 and 4

    :param vert_a:
    :param vert_d:
    :return:
    """

    vert_a_x, vert_a_y = vert_a
    vert_d_x, vert_d_y = vert_d
    len_w = abs(vert_d_x - vert_a_x)
    len_h = abs(vert_d_y - vert_a_y)
    a_rect = abs(len_w * len_h)
    return a_rect

def circular_aperture(arrs, circle_loc, rcirc, normalize=True, plot=False, plot_marker_size=500):
    """
    The main function to compute the fraction of an
    array of pixels which fall within a circular aperture

    :param arrs:
    :param circle_loc:
    :param rcirc:
    :param normalize:
    :param plot:
    :param plot_marker_size:
    :return:
    """
    square_size = plot_marker_size
    xarr, yarr = arrs
    offsetx, offsety = circle_loc
    sizex, sizey = xarr.shape

    # In this function the circle is always centred on (0,0)
    # so we translate the pixel coords such that the aperture location is at (0,0)
    xarr -= offsetx
    yarr -= offsety

    # Determine the size of each pixel
    pix_sizex = xarr[1, 0] - xarr[0, 0]
    pix_sizey = yarr[0, 1] - yarr[0, 0]

    # Determine the vertices of the pixels
    vert1x = xarr - pix_sizex / 2.
    vert1y = yarr + pix_sizey / 2.  # 1 _________ 4
    vert2x = xarr - pix_sizex / 2.
    vert2y = yarr - pix_sizey / 2.  # |		      |
    vert3x = xarr + pix_sizex / 2.
    vert3y = yarr - pix_sizey / 2.  # |	    '	  |
    vert4x = xarr + pix_sizex / 2.
    vert4y = yarr + pix_sizey / 2.  # 2 --------- 3

    # Determine the radial distance to each pixel vertex from the centre of the aperture (at (0,0))
    rvert1 = np.sqrt(vert1x ** 2 + vert1y ** 2)
    rvert2 = np.sqrt(vert2x ** 2 + vert2y ** 2)
    rvert3 = np.sqrt(vert3x ** 2 + vert3y ** 2)
    rvert4 = np.sqrt(vert4x ** 2 + vert4y ** 2)

    #     +++++_____
    #   +     | +   |
    #  +      |  +  |
    #  +       --+--
    #   +       +
    #     +++++

    # Find the vertices that are nearest and farthest from the aperture centre
    biggest_vertx = np.zeros_like(xarr)
    biggest_verty = np.zeros_like(yarr)
    smallest_vertx = np.zeros_like(xarr)
    smallest_verty = np.zeros_like(yarr)
    for i in xrange(sizex):
        for j in xrange(sizey):
            x_i = xarr[i, j]
            y_j = yarr[i, j]
            if x_i >= 0:
                if y_j >= 0:
                    # then 4 for biggest
                    biggest_vertx[i, j] = vert4x[i, j]
                    biggest_verty[i, j] = vert4y[i, j]
                    # then 2 for smallest
                    smallest_vertx[i, j] = vert2x[i, j]
                    smallest_verty[i, j] = vert2y[i, j]
                elif y_j < 0:
                    # then 3 for biggest
                    biggest_vertx[i, j] = vert3x[i, j]
                    biggest_verty[i, j] = vert3y[i, j]
                    # then 1 for smallest
                    smallest_vertx[i, j] = vert1x[i, j]
                    smallest_verty[i, j] = vert1y[i, j]
            elif x_i < 0:
                if y_j > 0:
                    # then 1 for biggest
                    biggest_vertx[i, j] = vert1x[i, j]
                    biggest_verty[i, j] = vert1y[i, j]
                    # then 3 for smallest
                    smallest_vertx[i, j] = vert3x[i, j]
                    smallest_verty[i, j] = vert3y[i, j]
                elif y_j < 0:
                    # then 2 for biggest
                    biggest_vertx[i, j] = vert2x[i, j]
                    biggest_verty[i, j] = vert2y[i, j]
                    # then 4 for smallest
                    smallest_vertx[i, j] = vert4x[i, j]
                    smallest_verty[i, j] = vert4y[i, j]

    # Determine the radial distance to each pixel from the aperture
    rarr = np.sqrt(xarr ** 2 + yarr ** 2)
    rarr_shape = rarr.shape
    # Find the largest and smallest radial distances for each pixel,
    # which depends on which of the vertices are closest and furthest
    # from the aperture centre
    biggest_rarr = np.sqrt(biggest_vertx ** 2 + biggest_verty ** 2)
    smallest_rarr = np.sqrt(smallest_vertx ** 2 + smallest_verty ** 2)

    # Prepare some arrays to put the various pixels in, depending on
    # whether they're inside, crossing or outside the aperture
    crossing_arr = rarr.copy()
    outside_arr = rarr.copy()
    inside_arr = rarr.copy()
    # This case deals with where the aperture edge crosses many pixels,
    # and where the aperture is small and sits on the intersection of four pixels
    if ((pix_sizex < rcirc and pix_sizey < rcirc) or not
        ((rvert1.any() < rcirc) or (rvert2.any() < rcirc) or
         (rvert3.any() < rcirc) or (rvert4.any() < rcirc))):
        print "The circle is bigger than the pixels and/or covers more than two "
        outer_inside = np.where(biggest_rarr < rcirc, 1., np.nan)
        outer_outside = np.where(biggest_rarr > rcirc, 1., np.nan)
        inner_inside = np.where(smallest_rarr < rcirc, 1., np.nan)
        inner_outside = np.where(smallest_rarr > rcirc, 1., np.nan)
        # Everything that touches the circle
        crossing_arr *= inner_inside
        crossing_arr *= outer_outside
        # everything totally outside the circle
        outside_arr *= inner_outside
        # everything totally inside the circle
        inside_arr += outer_inside

        to_do = np.where(np.isnan(crossing_arr) == False)
        inside_mask = outer_inside.copy()
        outside_mask = inner_outside.copy()
    # This case handles where:
    # - The aperture sits over two pixels, crossing a vertical line;
    # - The aperture sits over two pixels, crossing a horizontal line;
    # - The aperture sits inside one pixel and crosses no lines
    else:
        # These tests determine whether the aperture crosses a vertical or horizontal line
        test12s = []
        test34s = []
        test23s = []
        test14s = []
        for i in xrange(xarr.shape[0] * yarr.shape[0]):
            # If there's any intersects with vertical pixel edges..
            int12x_pos, int12x_neg, int12y_pos, int12y_neg, test12 = \
                intersect([vert1x.flatten()[i], vert1y.flatten()[i]],
                          [vert2x.flatten()[i], vert2y.flatten()[i]],
                          rcirc)
            test12s.append(test12)

            int34x_pos, int34x_neg, int34y_pos, int34y_neg, test34 = \
                intersect([vert3x.flatten()[i], vert3y.flatten()[i]],
                          [vert4x.flatten()[i], vert4y.flatten()[i]],
                          rcirc)
            test34s.append(test34)

            # If there's any intersects with horizontal pixel edges..
            int23x_pos, int23x_neg, int23y_pos, int23y_neg, test23 = \
                intersect([vert2x.flatten()[i], vert2y.flatten()[i]],
                          [vert3x.flatten()[i], vert3y.flatten()[i]],
                          rcirc)
            test23s.append(test23)

            int14x_pos, int14x_neg, int14y_pos, int14y_neg, test14 = \
                intersect([vert1x.flatten()[i], vert1y.flatten()[i]],
                          [vert4x.flatten()[i], vert4y.flatten()[i]],
                          rcirc)
            test14s.append(test14)

        test12s = np.array(test12s)
        test34s = np.array(test34s)
        test23s = np.array(test23s)
        test14s = np.array(test14s)

        if test12s.any() or test34s.any():
            # All the elements of row returned by the_row_loc will be the same
            # (cos if the way xarr and yarr are made - mesgrid style),
            # so we just grab any two (the first two in this case)
            the_row_loc = np.where((vert1y.flatten() > 0) & (vert2y.flatten() < 0))[0][:2]
            # the_row = [vert1y.flatten()[the_row_loc], vert2y.flatten()[the_row_loc]]
            # We want the two pix closest to the circle (centred on (0,0))
            # So we sort the x coords from small to large (keepign track of index with enumerate)
            # and grab the two smallset (being the closest to zero)
            sorted_pix = sorted(enumerate(np.absolute(xarr[:, 0])), key=lambda x: x[1])
            # Grab the locations of the nearest two pixels
            the_col_locs = list(zip(*sorted_pix[:2])[0])
            print "\nThe circle covers two pixels, and crosses a vertical line\n"
            to_do = (np.array(the_col_locs), np.array(the_row_loc))
            # yarr goes col by row, so to the the row no. we go mod(flattened index,xshape of arr)
            # e.g. [3%6, 9%6] = [3,3] --> 4th row (indexing from zero)
            to_do = ([to_do_i for to_do_i in to_do[0]],
                     [to_do_i % rarr.shape[1] for to_do_i in to_do[1]])

        elif test23s.any() or test14s.any():
            the_col_loc = np.where((vert1x.flatten() < 0) & (vert4x.flatten() > 0))[0][:2]
            # the_col = [vert1x.flatten()[the_col_loc], vert4x.flatten()[the_col_loc]]
            print "\nThe circle covers two pixels, and crosses a horizontal line\n"
            # We want the two pix closest to the circle (centred on (0,0))
            # So we sort the x coords from small to large (keepign track of index with enumerate)
            # and grab the two smallset (being the closest to zero)
            sorted_pix = sorted(enumerate(np.absolute(yarr[0, :])), key=lambda y: y[1])
            # Grab the locations of the nearest two pixels
            the_row_locs = list(zip(*sorted_pix[:2])[0])
            to_do = (np.array(the_col_loc), np.array(the_row_locs))
            # xarr goes row by col, so to the the row no. we go int(flattened index/xshape of arr)
            # e.g. [18./6, 19./6] = [3.0,3.166666..], whereas [int(18/6), int(19/6)] = [3,3]
            # --> 4th col. (indexing from 0)
            to_do = ([int(to_do_i / rarr.shape[0]) for to_do_i in to_do[0]],
                     [to_do_i for to_do_i in to_do[1]])
            # print the_row, the_pix_locs, xarr[:,0].flatten()[the_pix_locs]

        else:
            the_pix_loc = np.where((vert1y.flatten() > 0) &
                                   (vert2y.flatten() < 0) &
                                   (vert1x.flatten() < 0) &
                                   (vert4x.flatten() > 0))
            to_do = the_pix_loc
            # the_pix = [xarr.flatten()[the_pix_loc], xarr.flatten()[the_pix_loc]]
            to_do = ([int(to_do_i / rarr.shape[0]) for to_do_i in to_do[0]],
                     [to_do_i % rarr.shape[1] for to_do_i in to_do[0]])
            print "\nThe circle is smaller than the pixels and doesn't cross any of them!\n"

        # Create the three masks for this case
        outside_mask = np.ones_like(rarr)
        outside_mask[to_do] = np.nan

        inside_mask = np.ones_like(rarr) * np.nan

        crossing_mask = inside_mask.copy()
        crossing_mask[to_do] = 1.0

        # Reshape the flattened arrays back to their original shape
        crossing_mask = np.reshape(crossing_mask, rarr_shape)
        outside_mask = np.reshape(outside_mask, rarr_shape)
        inside_mask = np.reshape(inside_mask, rarr_shape)

        # Mask them using the masks created above
        crossing_arr *= crossing_mask
        outside_arr *= outside_mask
        inside_arr *= inside_mask

    #################################

    # plt.clf()
    # plt.close()
    # fig = plt.figure()
    # axis = fig.add_subplot(111, aspect='equal')
    # axis.scatter(xarr[to_do] + offsetx, yarr[to_do] + offsety, 200, c='g', marker='o')
    # axis.scatter(xarr + offsetx, yarr + offsety, square_size, c='k', marker='s')
    # axis.scatter(xarr + offsetx, yarr + offsety, 200, c='k', marker='.')
    # circ = plt.Circle([offsetx, offsety], rcirc, color='r', linewidth=3, fill=None)
    # axis.add_patch(circ)
    # plt.savefig('debug.png')

    #################################

    # Use the masks to determine which pixles need to be treated which way
    inside_todo = np.where(inside_mask == 1.)
    inside_arr_todo = inside_arr[inside_todo]
    x_inside_todo = xarr[inside_todo]
    y_inside_todo = yarr[inside_todo]

    outside_todo = np.where(outside_mask == 1.)
    outside_arr_todo = inside_arr[outside_todo]
    x_outside_todo = xarr[outside_todo]
    y_outside_todo = yarr[outside_todo]

    # Obviously the most complicated case is where the aperture edge crosses the pixels...
    crossing_arr_todo = crossing_arr[to_do]

    # We need to array entries...
    xarr_todo = xarr[to_do]
    yarr_todo = yarr[to_do]

    # and the vertex entries.
    vert1x_todo = vert1x[to_do]
    vert1y_todo = vert1y[to_do]
    vert1r_todo = np.sqrt(vert1x_todo ** 2 + vert1y_todo ** 2)

    vert2x_todo = vert2x[to_do]
    vert2y_todo = vert2y[to_do]
    vert2r_todo = np.sqrt(vert2x_todo ** 2 + vert2y_todo ** 2)

    vert3x_todo = vert3x[to_do]
    vert3y_todo = vert3y[to_do]
    vert3r_todo = np.sqrt(vert3x_todo ** 2 + vert3y_todo ** 2)

    vert4x_todo = vert4x[to_do]
    vert4y_todo = vert4y[to_do]
    vert4r_todo = np.sqrt(vert4x_todo ** 2 + vert4y_todo ** 2)

    # 1 _________ 4
    #  |		 |
    #  |	'	 |
    # 2 --------- 3
    if plot:
        # If a plot is required, create the axes and start plotting here
        plt.clf()
        plt.close()
        fig = plt.figure()
        axis = fig.add_subplot(111, aspect='equal')
        axis.scatter(xarr + offsetx, yarr + offsety, square_size, c='w', marker='s')

    # Now we do the complicated bit: Individually computing the different ways
    #  in which a circle can cross a square!
    output = [[], [], [], []]
    for i in xrange(crossing_arr_todo.shape[0]):

        # Determine which pixel vertices are inside and outside the aperture
        vert1_outside = (vert1r_todo[i] > rcirc)
        vert2_outside = (vert2r_todo[i] > rcirc)
        vert3_outside = (vert3r_todo[i] > rcirc)
        vert4_outside = (vert4r_todo[i] > rcirc)

        vert1_inside = not vert1_outside
        vert2_inside = not vert2_outside
        vert3_inside = not vert3_outside
        vert4_inside = not vert4_outside


        # 1 _________ 4
        #  |		 |
        #  |	'	 |
        # 2 --------- 3

        # Compute the pixel-circle intercepts
        int12x_pos, int12x_neg, int12y_pos, int12y_neg, test12\
            = intersect([vert1x_todo[i], vert1y_todo[i]],
                        [vert2x_todo[i], vert2y_todo[i]], rcirc)
        int23x_pos, int23x_neg, int23y_pos, int23y_neg, test23\
            = intersect([vert2x_todo[i], vert2y_todo[i]],
                        [vert3x_todo[i], vert3y_todo[i]], rcirc)
        int34x_pos, int34x_neg, int34y_pos, int34y_neg, test34\
            = intersect([vert3x_todo[i], vert3y_todo[i]],
                        [vert4x_todo[i], vert4y_todo[i]], rcirc)
        int41x_pos, int41x_neg, int41y_pos, int41y_neg, test41\
            = intersect([vert4x_todo[i], vert4y_todo[i]],
                        [vert1x_todo[i], vert1y_todo[i]], rcirc)

        intersects = [[int12x_pos, int12x_neg, int12y_pos, int12y_neg],
                      [int23x_pos, int23x_neg, int23y_pos, int23y_neg],
                      [int34x_pos, int34x_neg, int34y_pos, int34y_neg],
                      [int41x_pos, int41x_neg, int41y_pos, int41y_neg]]

        # Check each possible scenario....
        if vert1_outside and vert2_inside and vert3_inside and vert4_inside:
            # print 'case1'
            a_sm = circular_seg([int12x_pos, int12y_pos], [int41x_neg, int41y_neg], rcirc)
            a_tr = triangle([int12x_pos, int12y_pos], [int41x_neg, int41y_neg],
                            [vert1x_todo[i], vert1y_todo[i]])
            a_rect = rectangle([vert1x_todo[i], vert1y_todo[i]], [vert3x_todo[i], vert3y_todo[i]])
            area = a_rect - a_tr + a_sm
            #      + + + +
            #   1+_______4 +
            #   +        |   +
            #  +|        |    +
            #  +2________3    +
            #   +            +
            #    +          +
            #	   + + + +
            # --> Circle segment and outside triangle and area of pixel
        elif vert2_outside and vert1_inside and vert3_inside and vert4_inside:
            # print 'case2'
            a_sm = circular_seg([int12x_neg, int12y_neg], [int23x_neg, int23y_neg], rcirc)
            a_tr = triangle([int12x_neg, int12y_neg], [int23x_neg, int23y_neg],
                            [vert2x_todo[i], vert2y_todo[i]])
            a_rect = rectangle([vert1x_todo[i], vert1y_todo[i]], [vert3x_todo[i], vert3y_todo[i]])
            area = a_rect - a_tr + a_sm
            #      + + + +
            #    +         +
            #  +            +
            # + 1_______4    +
            # + |       |    +
            #  +|       |   +
            #   2+______3  +
            #	   + + + +
            # --> Circle segment and outside triangle and area of pixel
        elif vert3_outside and vert1_inside and vert2_inside and vert4_inside:
            # print 'case3'
            a_sm = circular_seg([int23x_pos, int23y_pos], [int34x_pos, int34y_neg], rcirc)
            a_tr = triangle([int23x_pos, int23y_pos], [int34x_pos, int34y_neg],
                            [vert3x_todo[i], vert3y_todo[i]])
            a_rect = rectangle([vert1x_todo[i], vert1y_todo[i]], [vert3x_todo[i], vert3y_todo[i]])
            area = a_rect - a_tr + a_sm
            #      + + + +
            #    +          +
            #   +            +
            #  +    1________4+
            #  +    |        |+
            #   +   |        +
            #    +  2______+_3
            #	   + + + +
            # --> Circle segment and outside triangle and area of pixel
        elif vert4_outside and vert1_inside and vert2_inside and vert3_inside:
            # print 'case4'
            a_sm = circular_seg([int41x_pos, int41y_pos], [int34x_pos, int34y_pos], rcirc)
            a_tr = triangle([int41x_pos, int41y_pos], [int34x_pos, int34y_pos],
                            [vert4x_todo[i], vert4y_todo[i]])
            a_rect = rectangle([vert1x_todo[i], vert1y_todo[i]], [vert3x_todo[i], vert3y_todo[i]])
            area = a_rect - a_tr + a_sm
            #      + + + +
            #    +  1_______+4
            #   +   |        +
            #  +    |        |+
            #  +    2________3+
            #   +            +
            #    +          +
            #	   + + + +
            # --> Circle segment and outside triangle and area of pixel
        elif vert2_outside and vert3_outside and vert1_inside and vert4_inside:
            # print 'case5'
            a_sm = circular_seg([int34x_pos, int34y_neg], [int12x_pos, int12y_neg], rcirc)
            a_tr1 = triangle([int34x_pos, int34y_neg], [int12x_pos, int12y_neg],
                             [vert4x_todo[i], vert4y_todo[i]])
            a_tr2 = triangle([vert1x_todo[i], vert1y_todo[i]], [int12x_pos, int12y_neg],
                             [vert4x_todo[i], vert4y_todo[i]])
            area = a_sm + a_tr1 + a_tr2
            #      + + + +
            #    +         +
            #   +            +
            #  +  1________4  +
            #  +  |        |  +
            #   + |        | +
            #    +|        |+
            #	  |+ + + + |
            #     2________3
            # --> Circle segment and two triangles
        elif vert3_outside and vert4_outside and vert1_inside and vert2_inside:
            # print 'case6'
            a_sm = circular_seg([int41x_pos, int41y_pos], [int23x_pos, int23y_pos], rcirc)
            a_tr1 = triangle([int41x_pos, int41y_pos],
                             [int23x_pos, int23y_pos], [vert2x_todo[i], vert2y_todo[i]])
            a_tr2 = triangle([vert1x_todo[i], vert1y_todo[i]], [int41x_pos, int41y_pos],
                             [vert2x_todo[i], vert2y_todo[i]])
            area = a_sm + a_tr1 + a_tr2
            #      + + + +
            #    +          +
            #   +       1____+___4
            #  +        |     +  |
            #  +        |     +  |
            #   +       |    +   |
            #    +      2___+____3
            #	   + + + +
            # --> Circle segment and two triangles
        elif vert2_outside and vert3_outside and vert4_outside and vert1_inside:
            # print 'case7'
            a_sm = circular_seg([int41x_pos, int41y_pos], [int12x_pos, int12y_neg], rcirc)
            a_tr = triangle([int41x_pos, int41y_pos],
                            [int12x_pos, int12y_neg], [vert1x_todo[i], vert1y_todo[i]])
            area = a_sm + a_tr
            #     +++++
            #   +       +
            #  +      1__+____4
            #  +      |  +    |
            #   +     | +     |
            #     +++++       |
            #		  2_______3
            # --> Circle segment and triangle
        elif vert1_outside and vert4_outside and vert2_inside and vert3_inside:
            # print 'case8'
            a_sm = circular_seg([int12x_pos, int12y_pos], [int34x_pos, int34y_pos], rcirc)
            a_tr1 = triangle([int12x_pos, int12y_pos],
                             [int34x_pos, int34y_pos], [vert2x_todo[i], vert2y_todo[i]])
            a_tr2 = triangle([vert3x_todo[i], vert3y_todo[i]], [int34x_pos, int34y_pos],
                             [vert2x_todo[i], vert2y_todo[i]])
            area = a_sm + a_tr1 + a_tr2
            #     1________4
            #     |        |
            #     |+ + + + |
            #    +|        |+
            #   + 2________3 +
            #  +              +
            #  +              +
            #   +            +
            #    +         +
            #	   + + + +
            # --> Circle segment and two triangles
        elif vert1_outside and vert3_outside and vert4_outside and vert2_inside:
            # print 'case9'
            a_sm = circular_seg([int23x_pos, int23y_pos], [int12x_pos, int12y_pos], rcirc)
            a_tr = triangle([int23x_pos, int23y_pos],
                            [int12x_pos, int12y_pos], [vert2x_todo[i], vert2y_todo[i]])
            area = a_sm + a_tr
            #		  1_______4
            #		  |       |
            #     +++++       |
            #   +     | +     |
            #  +      2__+____3
            #  +         +
            #   +       +
            #     +++++
            # --> Circle segment and triangle
        elif vert1_outside and vert2_outside and vert3_outside and vert4_outside and test12:
            # print 'case10'
            a_sm = circular_seg([int12x_pos, int12y_pos], [int12x_pos, int12y_neg], rcirc)
            area = a_sm
            #		  1________________4
            #		  |                |
            #     +++++                |
            #   +     | +              |
            #  +      |  +             |
            #  +      |  +             |
            #   +     | +              |
            #     +++++                |
            #		  |                |
            #		  2________________3
            # --> Circle segment
        elif vert1_outside and vert2_outside and vert3_inside and vert4_inside:
            # print 'case11'
            a_sm = circular_seg([int41x_neg, int41y_pos], [int23x_neg, int23y_pos], rcirc)
            a_tr1 = triangle([int41x_neg, int41y_pos],
                             [int23x_neg, int23y_pos], [vert3x_todo[i], vert3y_todo[i]])
            a_tr2 = triangle([vert4x_todo[i], vert4y_todo[i]], [int41x_neg, int41y_pos],
                             [vert3x_todo[i], vert3y_todo[i]])
            area = a_sm + a_tr1 + a_tr2
            #          + + +
            #  1 ___+___4    +
            #  |   +    |     +
            #  |   +    |     +
            #  2___+____3     +
            #       +        +
            #         + + +
            # --> Circle segment and two triangles
        elif vert1_outside and vert2_outside and vert4_outside and vert3_inside:
            # print 'case12'
            a_sm = circular_seg([int23x_neg, int23y_pos], [int34x_pos, int34y_pos], rcirc)
            a_tr = triangle([int23x_neg, int23y_pos],
                            [int34x_pos, int34y_pos], [vert3x_todo[i], vert3y_todo[i]])
            area = a_sm + a_tr
            #  1______4
            #  |      |
            #  |      +++++
            #  |    + |     +
            #  2___+__3      +
            #      +         +
            #       +       +
            #         +++++
            # --> Circle segment and triangle
        elif vert1_outside and vert2_outside and vert3_outside and vert4_outside and test23:
            # print 'case13'
            a_sm = circular_seg([int23x_pos, int23y_pos], [int23x_neg, int23y_pos], rcirc)
            area = a_sm
            # 1____________4
            # |			   |
            # |	           |
            # |	           |
            # |   +++++    |
            # 2_+_______+__3
            #  +         +
            #  +         +
            #   +       +
            #     +++++
            # --> Circle segment
        elif vert1_outside and vert2_outside and vert3_outside and vert4_inside:
            # print 'case14'
            a_sm = circular_seg([int41x_neg, int41y_pos], [int34x_pos, int34y_neg], rcirc)
            a_tr = triangle([int41x_neg, int41y_pos],
                            [int34x_pos, int34y_neg], [vert4x_todo[i], vert4y_todo[i]])
            area = a_sm + a_tr
            #       + + +
            #      +     +
            # 1___+___4   +
            # |   +   |   +
            # |    +  |  +
            # |     ++++
            # |       |
            # 2_______3
            # --> Circle segment and triangle
        elif vert1_outside and vert2_outside and vert3_outside and vert4_outside and test34:
            # print 'case15'
            a_sm = circular_seg([int34x_pos, int34y_pos], [int34x_pos, int34y_neg], rcirc)
            area = a_sm
            # 1_____________4
            # |             |
            # |          + + +
            # |         +   |  +
            # |        +    |   +
            # |        +    |   +
            # |         +   |  +
            # |           ++++
            # 2_____________3
            # --> Circle segment
        elif vert1_outside and vert2_outside and vert3_outside and vert4_outside and test41:
            # print 'case16'
            a_sm = circular_seg([int41x_pos, int41y_pos], [int41x_neg, int41y_pos], rcirc)
            area = a_sm
            #       ++++
            #     +      +
            #    +        +
            # 1__+________+_4
            # |   +      +  |
            # |     ++++    |
            # |             |
            # |             |
            # |             |
            # 2_____________3
            # --> Circle segment
        elif (vert1_outside and vert2_outside and vert3_outside and vert4_outside and not
             test12 and not test23 and not test34 and not test41):
            # print 'case17'
            a_circ = np.pi * (rcirc ** 2)
            area = a_circ
            # 1_____________4
            # |     ++++    |
            # |   +      +  |
            # |  +        + |
            # |  +        + |
            # |   +      +  |
            # |     ++++    |
            # 2_____________3
            # --> Circle segment
        else:
            raise NameError(
                "No area has been defined - something has gone wrong with the code. "
                "Please contact the developer."
            )
        # For each pixel, after the appropriate case has been determined
        # and the area computed, put it all into a nested list
        output[0].append(xarr_todo[i])
        output[1].append(yarr_todo[i])
        output[2].append(area)
        output[3].append(intersects)

    # Then do the much much simpler cases of the inner pixels (so we keep the whole area)...
    inside_aa = []
    for i in xrange(inside_arr_todo.shape[0]):
        area = pix_sizex * pix_sizey
        inside_aa.append(area)
    # And the pixels outside (so the area is always 0)
    outside_aa = [0 for ii in outside_arr_todo]

    # Shift the arrays back to their original location for plotting
    x_coords, y_coords, area_at_coord, ints = output
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    x_coords += offsetx
    y_coords += offsety

    inside_x_coords = np.array(x_inside_todo)
    inside_y_coords = np.array(y_inside_todo)
    inside_x_coords += offsetx
    inside_y_coords += offsety

    outside_xx = np.array(x_outside_todo)
    outside_yy = np.array(y_outside_todo)
    outside_xx += offsetx
    outside_yy += offsety

    if plot:
        # Plotting the pixels that cross the circle
        for x_i, y_i, a_i, intersects in zip(x_coords, y_coords, area_at_coord, ints):
            axis.scatter(x_i, y_i, square_size, c=a_i, marker='s', alpha=1.,
                       vmax=pix_sizex * pix_sizey, vmin=0., linewidth=0)

        # Plotting the pixels fully inside the circle
        for x_i, y_i, a_i in zip(inside_x_coords, inside_y_coords, inside_aa):
            axis.scatter(x_i, y_i, square_size, c=a_i, marker='s', alpha=1.,
                       vmax=pix_sizex * pix_sizey, vmin=0., linewidth=0)

        # Plotting the pixels fully inside the circle
        for x_i, y_i, a_i in zip(outside_xx, outside_yy, outside_aa):
            axis.scatter(x_i, y_i, square_size, c=a_i, marker='s', alpha=1.,
                       vmax=pix_sizex * pix_sizey, vmin=0., linewidth=0)

        axis.scatter(xarr_todo + offsetx, yarr_todo + offsety, 200, c='k', marker='.')
        circ = plt.Circle([offsetx, offsety], rcirc, color='r', linewidth=3, fill=None)
        axis.add_patch(circ)

        # savedir = os.path.dirname(os.path.abspath(__file__))
        savedir = '.'
        saveloc = savedir+'/Aperture_fractions.png'
        print u"Saving plot to: '{0:s}'".format(saveloc)
        plt.savefig(saveloc)

    # Join all the arrays back together
    all_x = list(x_coords)+list(outside_xx)+list(inside_x_coords)
    all_y = list(y_coords)+list(outside_yy)+list(inside_y_coords)
    all_a = list(area_at_coord)+list(outside_aa)+list(inside_aa)

    # If the fraction of the pixels is the desired output, normalise
    tot_pix_area = inside_aa[0] # Just any inside pixel... it gives the total area of one pixel
    if normalize:
        all_a = [all_Ai/tot_pix_area for all_Ai in all_a]

        # for x_i, y_i, a_i in zip(all_x, all_y, all_a):
        #     print x_i, y_i, a_i
    return all_x, all_y, all_a
