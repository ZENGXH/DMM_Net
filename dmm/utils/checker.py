import torch
import logging

def CHECK2D(tensor):
    assert(len(tensor.shape) == 2), 'get {} {}'.format(tensor.shape, len(tensor.shape))  
    return tensor.shape 

def CHECK4D(tensor):
    assert(len(tensor.shape) == 4), 'get {} {}'.format(tensor.shape, len(tensor.shape))
    return tensor.shape 

def CHECK5D(tensor):
    assert(len(tensor.shape) == 5), 'get {} {}'.format(tensor.shape, len(tensor.shape))
    return tensor.shape 

def CHECK3D(tensor):
    assert(len(tensor.shape) == 3), 'get {} {}'.format(tensor.shape, len(tensor.shape))
    return tensor.shape 

def CHECKSIZE(tensor, size):
    if isinstance(size, torch.Tensor):
        assert(tensor.shape == size.shape), ' get {} {}'.format(tensor.shape, size.shape)
    else:
        assert(tensor.shape == torch.Size(size)), tensor.shape

def CHECKEQ(a, b,s=None):
    assert(a == b), 'get {} {}'.format(a, b)
    if not (a == b):
        print('get {} {}'.format(a, b))
        if s:
            print(s)
        else: 
            assert(False)

def CHECKABSST(a, b):
    """ check if abs(a) < b """
    # assert(abs(a) < b), 'get {} {}'.format(a, b)
    if not (abs(a) < b):
        print('get {} {}'.format(a, b))

def CHECKEQT(a,b):
    CHECKEQ(a.shape, b.shape)
    CHECKEQ((a-b).sum(), 0)

def CHECKINFO(tensor,mode,info):
    for i in range(len(mode)):
        k = mode[i]
        if k.isdigit():
            v = int(k)
        else:
            v = info[k]
        assert(tensor.shape[i] == v), 'i{} k{} v{} get {}'.format(
                i,k,v,tensor.shape
                )
def CHECKBOX(a, b):
    a = a.cpu()
    b = b.cpu()
    assert(a.size == b.size), '{} {}'.format(a, b)
    CHECKEQ(a.bbox.shape, b.bbox.shape)
    CHECKEQ((a.bbox - b.bbox).sum(), 0)

def CHECKDEBUG(dmm_io, check_io, depth=0):
    """ check if every thing in A-list equal to B-list """
    if depth == 0 and type(dmm_io) == dict:
        return CHECKDEBUG([dmm_io], [check_io], depth=0)

    assert(type(dmm_io) == list)
    assert(type(check_io) == list)
    CHECKEQ(len(dmm_io), len(check_io))
    # logging.info(' '*(depth+1) + '>'*depth)
    for cid, (icur, iprev) in enumerate(zip(dmm_io, check_io)):
        depthstr = '   '*(depth+1) + '| ' + '[%d-%d]'%(depth, cid)
        logging.info(depthstr + '-')
        # logging.info(depthstr + 'cid: %d'%cid)
        if isinstance(icur, torch.Tensor):
            # logging.info(depthstr + 'check tensor: {}'.format(icur.shape))
            CHECKEQ(icur.shape, iprev.shape)
            CHECKEQ(icur.sum(), iprev.sum())
            # CHECKEQ(icur.mean(), iprev.mean())
            #CHECKEQ(icur.min(), iprev.min())
            #CHECKEQ(icur.max(), iprev.max())
        elif type(icur) == tuple:
            CHECKDEBUG(list(icur), list(iprev), depth+1)
        elif type(icur) == list:
            CHECKDEBUG(icur, iprev, depth+1) # list of list .. 
        elif type(icur) == dict:
            for name in icur.keys():
                logging.info(depthstr + 'dname: {}'.format(name))
                CHECKDEBUG([icur[name]], [iprev[name]], depth+1)
            #elif type(icur) == str:
            #    CHECKEQ(icur, iprev)
        else:
            logging.info(depthstr + '{}'.format(type(icur)))
            CHECKEQ(icur, iprev) # none special type
    # logging.info(' '*(depth+1) + '<'*depth)
    # logging.info(' ')
