# coding: utf8
from __future__ import unicode_literals

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293

NEGATE = \
["Aint", "arent", "no poder", "hipocresía", "no pudo", "darent", "didnt", "no",
 "no es", "no son", "hipocresía", "no pudo", "daren&#39;t", "no lo hizo", "no lo hace",
 "no", "hadnt", "hasnt", "havent", "no es", "poderoso", "mustnt", "ninguno",
 "no lo hagas", "no tenía", "no tiene", "no tiene", "no es", "podría no", "no debe",
 "necesidad", "no es necesario", "Nunca", "ninguna", "nope", "ni", "no", "nada", "en ninguna parte",
 "oughtnt", "Shant", "no deberia", "uhuh", "no fue", "werent",
 "no deberia", "no será", "no debería", "uh-uh", "no fue", "no fueron",
 "sin", "no", "no", "no lo hará", "no lo haría", "raramente", "raramente", "A pesar de"]

# booster/dampener 'intensifiers' or 'degree adverbs'
# http://en.wiktionary.org/wiki/Category:English_degree_adverbs

BOOSTER_DICT = \
{"absolutamente": B_INCR, "espantosamente": B_INCR, "muy": B_INCR, "completamente": B_INCR, "importantemente": B_INCR,
 "decididamente": B_INCR, "profundamente": B_INCR, "effing": B_INCR, "enormemente": B_INCR,
 "enteramente": B_INCR, "especialmente": B_INCR, "excepcionalmente": B_INCR, "extremadamente": B_INCR,
 "fabulosamente": B_INCR, "volteando": B_INCR, "flippin": B_INCR,
 "fricking": B_INCR, "frickin": B_INCR, "frigging": B_INCR, "friggin": B_INCR, "completamente": B_INCR, "maldito": B_INCR,
 "muy": B_INCR, "hella": B_INCR, "altamente": B_INCR, "enormemente": B_INCR, "increíblemente": B_INCR,
 "intensamente": B_INCR, "majorly": B_INCR, "Más": B_INCR, "más": B_INCR, "particularmente": B_INCR,
 "puramente": B_INCR, "bastante": B_INCR, "De Verdad": B_INCR, "notablemente": B_INCR,
 "asi que": B_INCR, "sustancialmente": B_INCR,
 "a fondo": B_INCR, "totalmente": B_INCR, "tremendamente": B_INCR,
 "uber": B_INCR, "increíblemente": B_INCR, "extraordinariamente": B_INCR, "absolutamente": B_INCR,
 "muy": B_INCR,
 "casi": B_DECR, "apenas": B_DECR, "apenas": B_DECR, "just enough": B_DECR,
 "mas o menos": B_DECR, "un poco": B_DECR, "mas o menos": B_DECR, "mas o menos": B_DECR,
 "Menos": B_DECR, "pequeño": B_DECR, "ligeramente": B_DECR, "de vez en cuando": B_DECR, "parcialmente": B_DECR,
 "apenas": B_DECR, "ligeramente": B_DECR, "algo": B_DECR,
 "tipo de": B_DECR, "sorta": B_DECR, "sortof": B_DECR, "tipo de": B_DECR}

# check for special case idioms using a sentiment-laden keyword known to VADER
SPECIAL_CASE_IDIOMS = {"una mierda": 3, "la bomba": 3, "cabron": 1.5, "sí claro": -2,
                       "está a la altura": 2, "hijo de puta": -2}

OTHERS_DICT = {"NEVER": "nunca", "SO": "asi que", "THIS": "this", "KIND": "tipo",
               "OF": "de", "LEAST": "menos", "AT": "lo", "VERY": "muy"}

__all__ = ['NEGATE', 'BOOSTER_DICT', 'SPECIAL_CASE_IDIOMS', 'B_INCR', 'B_DECR', 'OTHERS_DICT']