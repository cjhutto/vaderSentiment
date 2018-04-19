# coding: utf-8
# Author: C.J. Hutto

import re


def get_list_from_file(file_name):
    """read the lines from a file into a list"""
    with open(file_name, mode='r', encoding='utf-8') as f1:
        lst = f1.readlines()
    return lst


def append_to_file(file_name, line_data):
    """append a line of text to a file"""
    with open(file_name, mode='a', encoding='utf-8') as f1:
        f1.write(line_data)
        f1.write("\n")


def squeeze_whitespace(text):
    """removes extra white space"""
    return re.sub(r'[\s]+', ' ', text)


def pad_ref(reference_code):
    return r'\U' + reference_code.zfill(8)


emoji_test_lst = get_list_from_file("emoji-test.txt")
unic_emoji_dict = {}
utf8_emoji_dict = {}

for line in emoji_test_lst:
    line = squeeze_whitespace(line.strip())
    if len(line) < 2 or line.startswith("#"):
        continue
    unicode_ref_lst = line[:line.find(";")].split()
    for i, ref in enumerate(unicode_ref_lst):
        ref = pad_ref(ref)
        unicode_ref_lst[i] = ref
    unicode_ref = ''.join(unicode_ref_lst)
    description = ' '.join(line[line.find("#"):].split()[2:])
    unic_emoji_dict[unicode_ref] = description
    append_to_file("emoji_unic_lexicon.txt", "{}\t{}".format(unicode_ref, description))
    utf8_ref = line[line.find("#"):].split()[1]
    utf8_emoji_dict[utf8_ref] = description
    append_to_file("emoji_utf8_lexicon.txt", "{}\t{}".format(utf8_ref, description))
print(unic_emoji_dict)
