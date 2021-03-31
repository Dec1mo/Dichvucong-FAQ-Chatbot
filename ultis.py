import re
import copy
import json

stop_words = ['bạn', 'ban', 'anh', 'chị', 'chi', 'em', 'shop', 'bot', 'ad']

def convert_to_no_accents(text):
    patterns = {
        '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
        '[đ]': 'd',
        '[èéẻẽẹêềếểễệ]': 'e',
        '[ìíỉĩị]': 'i',
        '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
        '[ùúủũụưừứửữự]': 'u',
        '[ỳýỷỹỵ]': 'y'
    }
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        output = re.sub(regex.upper(), replace.upper(), output)
    return output

def data_sampling():
    with open('data/data_dict.json', encoding='utf8') as json_file:
        data_dict = json.load(json_file)
    new_data_dict = copy.deepcopy(data_dict)

    j = 0
    for k, v in data_dict.items():
        a_question, _ = re.subn('\.{2,}', '.', v['question'].strip())
        if a_question[-1] == '.':
            a_question = a_question[:-1]
        dot_pos = a_question.find('.')
        if dot_pos != -1 and dot_pos != len(a_question) - 1:
            que_sens = a_question.split('.')
            real_que = ''
            for que in reversed(que_sens):
                if '?' in que:
                    real_que = que
            if not real_que:
                real_que = que_sens[-1]
            new_data_dict[j + len(data_dict)] = v
            new_data_dict[j + len(data_dict)]['question'] = real_que
            new_data_dict[j + len(data_dict)]['pid'] = k
            j += 1

    with open("data/new_data_dict.json", "w", encoding='utf8') as f:
        json.dump(new_data_dict, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    data_sampling()