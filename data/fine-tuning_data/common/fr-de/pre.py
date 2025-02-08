import re
import json
 
""" src_file = "/mnt/luoyingfeng/lora4mt/data/fine-tuning_data/common/fr-de/test.fr2de.fr"
tgt_file = "/mnt/luoyingfeng/lora4mt/data/fine-tuning_data/common/fr-de/test.fr2de.de"
src = [line.strip() for line in open(src_file,"r",encoding="utf-8").readlines()]
tgt = [line.strip() for line in open(tgt_file,"r",encoding="utf-8").readlines()]
assert len(src) == len(tgt)
with open("/mnt/luoyingfeng/lora4mt/data/fine-tuning_data/common/fr-de/test.fr2de.json" ,"w", encoding="utf-8") as file:
    for src_line,tgt_line in zip(src, tgt):
        my_dict = {"translation": {"fr": src_line ,"de": tgt_line}}
        json.dump(my_dict, file, ensure_ascii=False)
        file.write("\n") """

LANG_TABLE = {
    "af": "Afrikaans",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "av": "Avaric",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dz": "Dzongkha",
    "el": "Modern Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Gaelic",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Modern Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kirghiz",
    "li": "Limburgish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "nb": "Norwegian Bokmål",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "or": "Oriya",
    "pa": "Panjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "se": "Northern Sami",
    "sh": "Serbo-Croatian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovene",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tr": "Turkish",
    "tt": "Tatar",
    "ug": "Uighur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wa": "Walloon",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
    "zu": "Zulu",
}

import json
import os
lang_pair_list = ["de-cs","fr-de"]
for lang_pair in lang_pair_list:
    test1_file = "/mnt/luoyingfeng/lora4mt/data/fine-tuning_data/common/"+lang_pair+"/test."+lang_pair.split("-")[0]+"2"+lang_pair.split("-")[1]+".json"
    test2_file = "/mnt/luoyingfeng/lora4mt/data/fine-tuning_data/common/"+lang_pair+"/test."+lang_pair.split("-")[1]+"2"+lang_pair.split("-")[0]+".json"
    for file in [test1_file,test2_file]:
        data = [line.strip() for line in open(file).readlines()]
        tgt_path = "/mnt/luoyingfeng/lora4mt/data/fine-tuning_data/multi_llama_factory/"+file.split("/")[-1]
        with open(tgt_path,"w",encoding="utf-8") as wf:
            result = []
            for i in range(len(data)):
                line = json.loads(data[i])
                src_lang = LANG_TABLE[file.split("/")[-1].split(".")[1].split("2")[0]]
                tgt_lang = LANG_TABLE[file.split("/")[-1].split(".")[1].split("2")[1]]
                prompt = "Translate the following text from "+src_lang+" to "+tgt_lang+":\n"
                my_dict = {
                    "instruction": prompt + line["translation"][file.split("/")[-1].split(".")[1].split("2")[0]],
                    "input": "",
                    "output": line["translation"][file.split("/")[-1].split(".")[1].split("2")[1]]
                }
                result.append(my_dict)
                #my_dict = {"translation":json.loads(data[i])}
            json.dump(result, wf, indent=4, ensure_ascii=False)