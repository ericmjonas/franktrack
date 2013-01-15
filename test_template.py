from nose.tools import * 
import template
import numpy as np

def create_region(x, y):
    a = np.arange(x*y)
    a.shape = (y, x)
    return a
    
def test_overlap():
    ae = assert_equal
    # template / tgt same size
    ae(template.overlap(5, 5, 0), (0, 5))
    ae(template.overlap(5, 5, 1), (1, 5))
    ae(template.overlap(5, 5, 4), (4, 5))
    ae(template.overlap(5, 5, 5), (0, 0))

    ae(template.overlap(5, 5, -3), (0, 2))

    # check for smaller template
    ae(template.overlap(10, 5, 0), (0, 5))
    ae(template.overlap(10, 5, 1), (1, 6))
    ae(template.overlap(10, 5, 5), (5, 10))
    ae(template.overlap(10, 5, 10), (0, 0))
    
    # check for larger template
    ae(template.overlap(5, 10, 0), (0, 5))
    ae(template.overlap(5, 10, 1), (1, 5))
    ae(template.overlap(5, 10, 5), (0, 0))
    ae(template.overlap(5, 10, -5), (0, 5))
    
def test_template_select():
    # simple initial checks
    ae = assert_equal
    r1 = create_region(10, 5)
    t1 = create_region(3, 3)

    r1_a, t1_a = template.template_select(r1, t1, 0, 0)
    ae(r1_a.shape, (3, 3))
    ae(t1_a.shape, (3, 3))

    ae(r1_a[0, 0], 0)
    ae(r1_a[2, 2], 22)

    ae(t1_a[0, 0], 0)
    ae(t1_a[2, 2], 8)


    # simple offset
    r1 = create_region(10, 5)
    t1 = create_region(3, 3)
    r1_a, t1_a = template.template_select(r1, t1, 1, 2)
    ae(r1_a.shape, (3, 3))
    ae(t1_a.shape, (3, 3))

    ae(r1_a[0, 0], 21)
    ae(r1_a[2, 2], 43)

    ae(t1_a[0, 0], 0)
    ae(t1_a[2, 2], 8)

    # lower-right corner, almost
    r1 = create_region(10, 5)
    t1 = create_region(3, 3)
    r1_a, t1_a = template.template_select(r1, t1, 9, 3)
    ae(r1_a.shape, (2, 1))
    ae(t1_a.shape, (2, 1))

    ae(r1_a[0, 0], 39)
    ae(r1_a[1, 0], 49)

    ae(t1_a[0, 0], 0)
    ae(t1_a[1, 0], 3)

    # negative x-y spots
    r1 = create_region(10, 5)
    t1 = create_region(3, 4)
    r1_a, t1_a = template.template_select(r1, t1, -2, -1)
    ae(r1_a.shape, (3, 1))
    ae(t1_a.shape, (3, 1))

    ae(r1_a[0, 0], 0)
    ae(r1_a[2, 0], 20)

    ae(t1_a[0, 0], 5)
    ae(t1_a[2, 0], 11)
