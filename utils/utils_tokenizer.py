
import unicodedata



def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_end_of_word(text):
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


