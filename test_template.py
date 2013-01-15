from nose.tools import * 
import template

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
    
