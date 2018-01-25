# coding: utf8
from __future__ import unicode_literals

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293

NEGATE = \
["no és", "no arent", "no pot", "cant", "no podia", "atrevit", "no", "no",
 "no ho és", "no ho són", "no pot", "no podia", "no ho feu", "no ho va fer", "no ho fa",
 "no", "hadnt", "hasnt", "havent", "no", "potser", "no ha de", "cap dels dos",
 "no ho fas", "no ho havia fet", "no ho ha fet", "no ho han fet", "no ho és", "potser no", "no",
 "no necessito", "no necessites", "mai", "cap", "no", "ni", "no", "res", "enlloc",
 "no hauria de", "Shant", "no", "uhuh", "no era", "werent",
 "no hauria", "no ho faré", "no", "uh-uh", "no ho era", "no ho eren",
 "sense", "no wont", "no", "no ho farà", "no ho faria", "poques vegades", "rares vegades", "malgrat tot"]

# booster/dampener 'intensifiers' or 'degree adverbs'
# http://en.wiktionary.org/wiki/Category:English_degree_adverbs

BOOSTER_DICT = \
{"absolutament": B_INCR, "sorprenentment": B_INCR, "terriblement": B_INCR, "completament": B_INCR, "considerablement": B_INCR,
 "decididament": B_INCR, "profundament": B_INCR, "effing": B_INCR, "enormement": B_INCR,
 "completament": B_INCR, "especialment": B_INCR, "excepcionalment": B_INCR, "extremadament": B_INCR,
 "fabulós": B_INCR, "flipping": B_INCR, "flippin": B_INCR,
 "fricking": B_INCR, "frickin": B_INCR, "freda": B_INCR, "friggin": B_INCR, "completament": B_INCR, "follant": B_INCR,
 "en gran mesura": B_INCR, "hella": B_INCR, "molt": B_INCR, "moltíssim": B_INCR, "increïblement": B_INCR,
 "intensament": B_INCR, "Majorly": B_INCR, "més": B_INCR, "la majoria": B_INCR, "particularment": B_INCR,
 "purament": B_INCR, "bastant": B_INCR, "en realitat": B_INCR, "notablement": B_INCR,
 "tan": B_INCR, "substancialment": B_INCR,
 "a fons": B_INCR, "totalment": B_INCR, "tremendament": B_INCR,
 "uber": B_INCR, "increïblement": B_INCR, "inusual": B_INCR, "absolutament": B_INCR,
 "molt": B_INCR,
 "gairebé": B_DECR, "amb prou feines": B_DECR, "amb prou feines": B_DECR, "just enough": B_DECR,
 "tipus de": B_DECR, "una mica": B_DECR, "tipus de": B_DECR, "tipus de": B_DECR,
 "menys": B_DECR, "poc": B_DECR, "marginalment": B_DECR, "de tant en tant": B_DECR, "en part": B_DECR,
 "amb prou feines": B_DECR, "lleugerament": B_DECR, "una mica": B_DECR,
 "tipus de": B_DECR, "sorta": B_DECR, "sortof": B_DECR, "sort-of": B_DECR}

# check for special case idioms using a sentiment-laden keyword known to VADER
SPECIAL_CASE_IDIOMS = {"una merda": 3, "la bomba": 3, "cabró": 1.5, "sí, bé": -2,
                       "està a l'altura": 2, "fill de puta": -2}

OTHERS_DICT = {"NEVER": "mai", "SO": "tan", "THIS": "this", "KIND": "tipus",
               "OF": "de", "LEAST": "mínim", "AT": "a", "VERY": "molt"}

__all__ = ['NEGATE', 'BOOSTER_DICT', 'SPECIAL_CASE_IDIOMS', 'B_INCR', 'B_DECR', 'OTHERS_DICT']