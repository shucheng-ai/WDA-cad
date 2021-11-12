#!/usr/bin/env python3
import sysconfig
import sys
import os
CAD_CORE_LIB_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), 'build/lib.linux-x86_64-' + sys.version[:3]))
sys.path.append(CAD_CORE_LIB_PATH)

from cad_core import *

def dump_room (path, room):
    with open(path, 'w') as f:
        for obs in room:
            f.write('%d %d %d\n' % (obs['type'], int(obs['effective']), len(obs['polygon'])))
            for x, y in obs['polygon']:
                f.write('%d %d\n' % (x, y))
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    import beaver_core
    print(beaver_core.__file__)


