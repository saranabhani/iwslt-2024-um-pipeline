import string
import re
from camel_tools.utils.dediac import dediac_ar

ar_pattern = '[؟٪؛،a-zA-Z]'
def preprocess_text(txt, lang=eng):
	txt = txt.replace('M/', '')
    txt = txt.replace('O/', '')
    txt = txt.replace('U/', '')
    txt = txt.replace('UM/', '')
    txt = txt.replace('UO/', '')
    txt =txt.replace('/O', '')
    txt = txt.replace('%pw', '')
    txt = txt.replace('pw', '')
    txt = txt.replace('<non-MSA>', '')
    txt = txt.replace('</non-MSA>', '')
    txt = re.sub(r'<.*>', '', txt)
    if lang=='ar':
    	txt = dediac.dediac_ar(str(txt))
    	txt = re.sub(ar_pattern, '', txt)
    if lang=='mt':
    	translation_table = str.maketrans('', '', string.punctuation.replace('-', '').replace("'", "")) 
     	txt = txt.translate(translation_table) 
    else:
    	txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = re.sub(r'\s+', ' ', txt)
    txt = txt.lower().strip()
    return txt