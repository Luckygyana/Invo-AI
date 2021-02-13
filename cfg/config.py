
"""
Config file for the control f various params


"""
import os 
write_video = False
psm = 11
oem = 3

if os.name == 'nt':
    out_dir = r'./output'
else:
    out_dir = r'./output'
if os.name == 'nt':
    int_dir = r'./intermediate'
else:
    int_dir = r'./intermediate'
ROW_TOLERANCE = 20   ### (unit:px) tolerance limit in same row for top value for bbox)
COL_TOLERANCE = 20   ### (unit:px) tolerance limit in same COL for left value for bbox)
alphabet = {'alphabet' :'$%@.,&():ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/\\\n '}
TESS_CONFIG = '-c  tessedit_char_whitelist="$%@.-,&():ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/\\\n "  --psm {} --oem {}'.format(psm, oem)
if os.name == 'nt':
    TESS_EXEC = r'D:\Applications\Tesseract-OCR\tesseract.exe'
else:
    TESS_EXEC = r'/usr/bin/tesseract'