'''
A program to compute exactly what fraction of an
array of pixels falls inside a circular aperture.

Call as:
circ_app.circular_aperture(arrs, circle_loc, rcirc, normalize=True, plot=False, plot_marker_size=5000)

Inputs:
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

Returns:
- all_x: The x-coordinates of the pixels in a 1-D list
- all_y: The y-coordinates of the pixels in a 1-D list
- all_A: The Area or fraction of each pixel falling inside the aperture

'''
import numpy as np
import matplotlib.pyplot as plt
import os

__author__ = 'Catherine de Burgh-Day'


def rescale_array(arr):
    """ Rescale or normalise an array to be between 0 and 1"""
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


def intersect(vertA, vertB, circle_rad):
    # (based on http://mathworld.wolfram.com/Circle-LineIntersection.html)
    vertAx, vertAy = vertA
    vertBx, vertBy = vertB
    dx = vertBx - vertAx
    dy = vertBy - vertAy
    dr = np.sqrt(dx ** 2 + dy ** 2)
    D = vertAx * vertBy - vertBx * vertAy

    # Do this because np.sign(0.0) = 0.0 (not 1.0 or -1.0)
    sdy = dy
    if dy == 0.0:
        sdy = 1.
    sign_dy = np.sign(sdy)

    intx_1 = (1 / dr ** 2) * (D * dy - sign_dy * dx * np.sqrt((circle_rad ** 2) * (dr ** 2) - (D ** 2)))
    intx_2 = (1 / dr ** 2) * (D * dy + sign_dy * dx * np.sqrt((circle_rad ** 2) * (dr ** 2) - (D ** 2)))

    inty_1 = (1 / dr ** 2) * (-D * dx + abs(dy) * np.sqrt((circle_rad ** 2) * (dr ** 2) - (D ** 2)))
    inty_2 = (1 / dr ** 2) * (-D * dx - abs(dy) * np.sqrt((circle_rad ** 2) * (dr ** 2) - (D ** 2)))

    Delta = (circle_rad ** 2) * (dr ** 2) - (D ** 2)

    if Delta > 0:
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


def Circular_Seg(intA, intB, circle_rad):
    intAx, intAy = intA
    intBx, intBy = intB
    d = np.sqrt((intAx - intBx) ** 2 + (intAy - intBy) ** 2)
    angle = 2.0 * np.arcsin(0.5 * d / circle_rad)
    A_circ_seg = abs(0.5 * (circle_rad ** 2) * (angle - np.sin(angle)))
    return A_circ_seg


def Triangle(vertA, vertB, vertC):
    vertAx, vertAy = vertA
    vertBx, vertBy = vertB
    vertCx, vertCy = vertC
    A_tri = abs((vertAx * (vertBy - vertCy) +
                 vertBx * (vertCy - vertAy) +
                 vertCx * (vertAy - vertBy)) / 2.0)
    # Area = (1/2)b*h
    #      /\A
    #     /  \
    #    /    \
    #  B/______\C

    return A_tri


def Rectangle(vertA, vertD):
    # vert A and D must be opposite i.e. 1 and 3 or 2 and 4
    vertAx, vertAy = vertA
    vertDx, vertDy = vertD
    len_w = abs(vertDx - vertAx)
    len_h = abs(vertDy - vertAy)
    A_rect = abs(len_w * len_h)
    return A_rect

def circular_aperture(arrs, circle_loc, rcirc, normalize=True, plot=False, plot_marker_size=500):
    square_size=plot_marker_size
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
            xi = xarr[i, j]
            yj = yarr[i, j]
            if xi >= 0:
                if yj >= 0:
                    # then 4 for biggest
                    biggest_vertx[i, j] = vert4x[i, j]
                    biggest_verty[i, j] = vert4y[i, j]
                    # then 2 for smallest
                    smallest_vertx[i, j] = vert2x[i, j]
                    smallest_verty[i, j] = vert2y[i, j]
                elif yj < 0:
                    # then 3 for biggest
                    biggest_vertx[i, j] = vert3x[i, j]
                    biggest_verty[i, j] = vert3y[i, j]
                    # then 1 for smallest
                    smallest_vertx[i, j] = vert1x[i, j]
                    smallest_verty[i, j] = vert1y[i, j]
            elif xi < 0:
                if yj > 0:
                    # then 1 for biggest
                    biggest_vertx[i, j] = vert1x[i, j]
                    biggest_verty[i, j] = vert1y[i, j]
                    # then 3 for smallest
                    smallest_vertx[i, j] = vert3x[i, j]
                    smallest_verty[i, j] = vert3y[i, j]
                elif yj < 0:
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
    ((rvert1.any() < rcirc) or (rvert2.any() < rcirc) or (rvert3.any() < rcirc) or (rvert4.any() < rcirc))):
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
            to_do = ([to_do_i for to_do_i in to_do[0]], [to_do_i % rarr.shape[1] for to_do_i in to_do[1]])

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
            to_do = ([int(to_do_i / rarr.shape[0]) for to_do_i in to_do[0]], [to_do_i for to_do_i in to_do[1]])
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
    # ax = fig.add_subplot(111, aspect='equal')
    # ax.scatter(xarr[to_do] + offsetx, yarr[to_do] + offsety, 200, c='g', marker='o')
    # ax.scatter(xarr + offsetx, yarr + offsety, square_size, c='k', marker='s')
    # ax.scatter(xarr + offsetx, yarr + offsety, 200, c='k', marker='.')
    # circ = plt.Circle([offsetx, offsety], rcirc, color='r', linewidth=3, fill=None)
    # ax.add_patch(circ)
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
        ax = fig.add_subplot(111, aspect='equal')
        ax.scatter(xarr + offsetx, yarr + offsety, square_size, c='w', marker='s')

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
        int12x_pos, int12x_neg, int12y_pos, int12y_neg, test12 = intersect([vert1x_todo[i], vert1y_todo[i]],
                                                                           [vert2x_todo[i], vert2y_todo[i]],
                                                                           rcirc)
        int23x_pos, int23x_neg, int23y_pos, int23y_neg, test23 = intersect([vert2x_todo[i], vert2y_todo[i]],
                                                                           [vert3x_todo[i], vert3y_todo[i]],
                                                                           rcirc)
        int34x_pos, int34x_neg, int34y_pos, int34y_neg, test34 = intersect([vert3x_todo[i], vert3y_todo[i]],
                                                                           [vert4x_todo[i], vert4y_todo[i]],
                                                                           rcirc)
        int41x_pos, int41x_neg, int41y_pos, int41y_neg, test41 = intersect([vert4x_todo[i], vert4y_todo[i]],
                                                                           [vert1x_todo[i], vert1y_todo[i]],
                                                                           rcirc)
        intersects = [[int12x_pos, int12x_neg, int12y_pos, int12y_neg],
                      [int23x_pos, int23x_neg, int23y_pos, int23y_neg],
                      [int34x_pos, int34x_neg, int34y_pos, int34y_neg],
                      [int41x_pos, int41x_neg, int41y_pos, int41y_neg]]

        # Check each possible scenario....
        if vert1_outside and vert2_inside and vert3_inside and vert4_inside:
            # print 'case1'
            A_sm = Circular_Seg([int12x_pos, int12y_pos], [int41x_neg, int41y_neg], rcirc)
            A_tr = Triangle([int12x_pos, int12y_pos], [int41x_neg, int41y_neg],
                            [vert1x_todo[i], vert1y_todo[i]])
            A_rect = Rectangle([vert1x_todo[i], vert1y_todo[i]], [vert3x_todo[i], vert3y_todo[i]])
            Area = A_rect - A_tr + A_sm
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
            A_sm = Circular_Seg([int12x_neg, int12y_neg], [int23x_neg, int23y_neg], rcirc)
            A_tr = Triangle([int12x_neg, int12y_neg], [int23x_neg, int23y_neg],
                            [vert2x_todo[i], vert2y_todo[i]])
            A_rect = Rectangle([vert1x_todo[i], vert1y_todo[i]], [vert3x_todo[i], vert3y_todo[i]])
            Area = A_rect - A_tr + A_sm
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
            A_sm = Circular_Seg([int23x_pos, int23y_pos], [int34x_pos, int34y_neg], rcirc)
            A_tr = Triangle([int23x_pos, int23y_pos], [int34x_pos, int34y_neg],
                            [vert3x_todo[i], vert3y_todo[i]])
            A_rect = Rectangle([vert1x_todo[i], vert1y_todo[i]], [vert3x_todo[i], vert3y_todo[i]])
            Area = A_rect - A_tr + A_sm
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
            A_sm = Circular_Seg([int41x_pos, int41y_pos], [int34x_pos, int34y_pos], rcirc)
            A_tr = Triangle([int41x_pos, int41y_pos], [int34x_pos, int34y_pos],
                            [vert4x_todo[i], vert4y_todo[i]])
            A_rect = Rectangle([vert1x_todo[i], vert1y_todo[i]], [vert3x_todo[i], vert3y_todo[i]])
            Area = A_rect - A_tr + A_sm
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
            A_sm = Circular_Seg([int34x_pos, int34y_neg], [int12x_pos, int12y_neg], rcirc)
            A_tr1 = Triangle([int34x_pos, int34y_neg], [int12x_pos, int12y_neg],
                             [vert4x_todo[i], vert4y_todo[i]])
            A_tr2 = Triangle([vert1x_todo[i], vert1y_todo[i]], [int12x_pos, int12y_neg],
                             [vert4x_todo[i], vert4y_todo[i]])
            Area = A_sm + A_tr1 + A_tr2
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
            A_sm = Circular_Seg([int41x_pos, int41y_pos], [int23x_pos, int23y_pos], rcirc)
            A_tr1 = Triangle([int41x_pos, int41y_pos], [int23x_pos, int23y_pos], [vert2x_todo[i], vert2y_todo[i]])
            A_tr2 = Triangle([vert1x_todo[i], vert1y_todo[i]], [int41x_pos, int41y_pos],
                             [vert2x_todo[i], vert2y_todo[i]])
            Area = A_sm + A_tr1 + A_tr2
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
            A_sm = Circular_Seg([int41x_pos, int41y_pos], [int12x_pos, int12y_neg], rcirc)
            A_tr = Triangle([int41x_pos, int41y_pos], [int12x_pos, int12y_neg], [vert1x_todo[i], vert1y_todo[i]])
            Area = A_sm + A_tr
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
            A_sm = Circular_Seg([int12x_pos, int12y_pos], [int34x_pos, int34y_pos], rcirc)
            A_tr1 = Triangle([int12x_pos, int12y_pos], [int34x_pos, int34y_pos], [vert2x_todo[i], vert2y_todo[i]])
            A_tr2 = Triangle([vert3x_todo[i], vert3y_todo[i]], [int34x_pos, int34y_pos],
                             [vert2x_todo[i], vert2y_todo[i]])
            Area = A_sm + A_tr1 + A_tr2
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
            A_sm = Circular_Seg([int23x_pos, int23y_pos], [int12x_pos, int12y_pos], rcirc)
            A_tr = Triangle([int23x_pos, int23y_pos], [int12x_pos, int12y_pos], [vert2x_todo[i], vert2y_todo[i]])
            Area = A_sm + A_tr
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
            A_sm = Circular_Seg([int12x_pos, int12y_pos], [int12x_pos, int12y_neg], rcirc)
            Area = A_sm
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
            A_sm = Circular_Seg([int41x_neg, int41y_pos], [int23x_neg, int23y_pos], rcirc)
            A_tr1 = Triangle([int41x_neg, int41y_pos], [int23x_neg, int23y_pos], [vert3x_todo[i], vert3y_todo[i]])
            A_tr2 = Triangle([vert4x_todo[i], vert4y_todo[i]], [int41x_neg, int41y_pos],
                             [vert3x_todo[i], vert3y_todo[i]])
            Area = A_sm + A_tr1 + A_tr2
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
            A_sm = Circular_Seg([int23x_neg, int23y_pos], [int34x_pos, int34y_pos], rcirc)
            A_tr = Triangle([int23x_neg, int23y_pos], [int34x_pos, int34y_pos], [vert3x_todo[i], vert3y_todo[i]])
            Area = A_sm + A_tr
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
            A_sm = Circular_Seg([int23x_pos, int23y_pos], [int23x_neg, int23y_pos], rcirc)
            Area = A_sm
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
            A_sm = Circular_Seg([int41x_neg, int41y_pos], [int34x_pos, int34y_neg], rcirc)
            A_tr = Triangle([int41x_neg, int41y_pos], [int34x_pos, int34y_neg], [vert4x_todo[i], vert4y_todo[i]])
            Area = A_sm + A_tr
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
            A_sm = Circular_Seg([int34x_pos, int34y_pos], [int34x_pos, int34y_neg], rcirc)
            Area = A_sm
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
            A_sm = Circular_Seg([int41x_pos, int41y_pos], [int41x_neg, int41y_pos], rcirc)
            Area = A_sm
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
            A_circ = np.pi * (rcirc ** 2)
            Area = A_circ
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
        output[2].append(Area)
        output[3].append(intersects)

    # Then do the much much simpler cases of the inner pixels (so we keep the whole area)...
    inside_AA = []
    for i in xrange(inside_arr_todo.shape[0]):
        Area = pix_sizex * pix_sizey
        inside_AA.append(Area)
    # And the pixels outside (so the area is always 0)
    outside_AA = [0 for ii in outside_arr_todo]

    # Shift the arrays back to their original location for plotting
    xx, yy, AA, ints = output
    xx = np.array(xx)
    yy = np.array(yy)
    xx += offsetx
    yy += offsety

    inside_xx = np.array(x_inside_todo)
    inside_yy = np.array(y_inside_todo)
    inside_xx += offsetx
    inside_yy += offsety

    outside_xx = np.array(x_outside_todo)
    outside_yy = np.array(y_outside_todo)
    outside_xx += offsetx
    outside_yy += offsety

    if plot:
        # Plotting the pixels that cross the circle
        for xi, yi, Ai, intersects in zip(xx, yy, AA, ints):
            ax.scatter(xi, yi, square_size, c=Ai, marker='s', alpha=1.,
                       vmax=pix_sizex * pix_sizey, vmin=0., linewidth=0)

        # Plotting the pixels fully inside the circle
        for xi, yi, Ai in zip(inside_xx, inside_yy, inside_AA):
            ax.scatter(xi, yi, square_size, c=Ai, marker='s', alpha=1.,
                       vmax=pix_sizex * pix_sizey, vmin=0., linewidth=0)

        # Plotting the pixels fully inside the circle
        for xi, yi, Ai in zip(outside_xx, outside_yy, outside_AA):
            ax.scatter(xi, yi, square_size, c=Ai, marker='s', alpha=1.,
                       vmax=pix_sizex * pix_sizey, vmin=0., linewidth=0)

        ax.scatter(xarr_todo + offsetx, yarr_todo + offsety, 200, c='k', marker='.')
        circ = plt.Circle([offsetx, offsety], rcirc, color='r', linewidth=3, fill=None)
        ax.add_patch(circ)

        # savedir = os.path.dirname(os.path.abspath(__file__))
        savedir = '.'
        saveloc = savedir+'/Aperture_fractions.png'
        print u"Saving plot to: '{0:s}'".format(saveloc)
        plt.savefig(saveloc)

    # Join all the arrays back together
    all_x = list(xx)+list(outside_xx)+list(inside_xx)
    all_y = list(yy)+list(outside_yy)+list(inside_yy)
    all_A = list(AA)+list(outside_AA)+list(inside_AA)

    # If the fraction of the pixels is the desired output, normalise
    tot_pix_area = inside_AA[0] # Just any inside pixel... it gives the total area of one pixel
    if normalize:
        all_A = [all_Ai/tot_pix_area for all_Ai in all_A]

        # for xi, yi, Ai in zip(all_x, all_y, all_A):
        #     print xi, yi, Ai
    return all_x, all_y, all_A
