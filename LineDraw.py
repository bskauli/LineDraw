def sign(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    else :
        return -1


def discrete_line(p1,p2):
    """Given two points in a grid, returns a list of all grid cells meeting the line between the points"""
    xdiff = p2[0] - p1[0]
    ydiff = p2[1] - p1[1]
    xincrement = sign(xdiff)
    yincrement = sign(ydiff)

    i,j = (1,1)
    cells = [(i,j)]
    while abs(i)<abs(xdiff) or abs(j) < abs(ydiff):
        if abs(i*ydiff) <= abs(j*xdiff):
            i += xincrement
        else:
            j += yincrement
        cells.append((i,j))
    return cells



print(discrete_line((0,0),(-5,-9)))
