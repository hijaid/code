import json
import os
import random
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

PROMPT_TABLE={
    "Translate the following text from English to Chinese:\n":"将以下文本从英语翻译成中文:\n",
    "Translate the following text from Chinese to English:\n":"Translate the following text from Chinese to English:\n",
    "Translate the following text from English to German:\n":"Übersetzen Sie den folgenden Text von Englisch nach Deutsch:\n",
    "Translate the following text from German to English:\n":"Translate the following text from German to English:\n",
    "Translate the following text from English to Czech:\n":"Přeložte následující text z angličtiny do češtiny:\n",
    "Translate the following text from Czech to English:\n":"Translate the following text from Czech to English:\n",
    "Translate the following text from English to Russian:\n":"Переведите следующий текст с английского на русский:\n",
    "Translate the following text from Russian to English:\n":"Translate the following text from Russian to English:\n",
    "Translate the following text from English to Finnish:\n":"Käännä seuraava teksti englannista suomeksi:\n",
    "Translate the following text from Finnish to English:\n":"Translate the following text from Finnish to English:\n",
    "Translate the following text from English to Icelandic:\n":"Þýddu eftirfarandi texta frá ensku á íslensku:\n",
    "Translate the following text from Icelandic to English:\n":"Translate the following text from Icelandic to English:\n",
    "Translate the following text from English to Modern Hebrew:\n":"תרגם את הטקסט הבא מאנגלית לעברית מודרנית:\n",
    "Translate the following text from Modern Hebrew to English:\n":"Translate the following text from Modern Hebrew to English:\n",
}

lang_list = ["cs","de","ru","zh","is","fi","he"]
for lang in lang_list:
    for pair in [lang +"-en", "en-"+lang]:
        for file_set in ["valid","train","test."+pair.replace("-","2")]:
            result = []
            src_file = "/mnt/luoyingfeng/lora4mt/data/fine-tuning_data/common/"+lang+"-en/"+file_set+".json"
            src_lang = LANG_TABLE[pair.split("-")[0]]
            tgt_lang = LANG_TABLE[pair.split("-")[1]]
            en_prompt = "Translate the following text from "+src_lang+" to "+tgt_lang+":\n"
            prompt = PROMPT_TABLE[en_prompt]
            with open(src_file, 'r', encoding="utf-8") as file:
                data = file.readlines()
                for line in data:
                    result.append({
                        "instruction": prompt + json.loads(line)["translation"][pair.split("-")[0]],
                        "input": "",
                        "output": json.loads(line)["translation"][pair.split("-")[1]]
                    })
            print(len(result))
            tgt_file = "/mnt/luoyingfeng/lora4mt/data/fine-tuning_data/tgt_prompt/"+file_set.split(".")[0]+"."+pair+".json"
            with open(tgt_file, 'w') as file:
                json.dump(result, file, indent=4, ensure_ascii=False)

            # file_path = "/mnt/luoyingfeng/lora4mt/data/fine-tuning_data/tgt_prompt/"+file_set.split(".")[0]+"."+pair+".json"
            # with open(file_path, 'r', encoding="utf-8") as file:
            #     data = json.load(file)
            #     random.shuffle(data)
            # with open(file_path, 'w') as file:
            #     json.dump(data, file, indent=4, ensure_ascii=False)