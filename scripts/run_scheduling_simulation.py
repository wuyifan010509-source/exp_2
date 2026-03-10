#!/usr/bin/env python3
"""
快速运行人机调度仿真实验
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.scheduling_simulation.__main__ import main

if __name__ == "__main__":
    main()
