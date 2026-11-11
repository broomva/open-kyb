"""Sector and industry classification as a code predicate over the declared corporate object.

The policy names industry categories in English; packets declare their object in Spanish or
English. Each category the policy lists needs a lexicon entry here. A category the policy adds
without one cannot be matched, so the parser reports it as unrepresentable instead of silently
never firing.

A keyword match cannot read negation ("does not trade in weapons", "sin comercio de armas"),
exclusions ("other than real estate") or imitation ("toy pistols"). An activity that names a
category *and* carries such a cue is returned separately, and the rule reads it as `unknown`:
an officer decides, the lexicon never does. An activity that names no category reads as not
matching, so coverage matters: every corporate activity is a material fact the officer sees,
and misses are eval cases (suite 4).

Licensing is a fact the declared object does not establish. A match on a category the policy
qualifies as *unlicensed* is therefore `unknown` (an officer checks the licence), never `yes`.
"""

from __future__ import annotations

from collections.abc import Iterable

from kyb.policy.text import NEGATIONS, fold, matches

# Terms are matched on `fold`ed text at word boundaries (`text.matches`): a trailing "*" matches
# any word suffix, and "a & b" needs both parts within `text.NEAR` words. A leading "~" marks a
# weak term: it names the category only in some senses ("glue gun", "water pistol"), so on its
# own it reads `unknown`, never `yes`.
# Financing as an activity, never "financial": "export of financial software" is not trade finance.
# Every trade-finance term that names financing uses this, not a `financ*` stem.
_FINANCING = ("financing|finance|finances|financed|financiamiento*|financiacion*|financiar|"
              "financia|financian")  # fmt: skip
LEXICON: dict[str, tuple[str, ...]] = {
    "weapons and defense": ("weapon*", "firearm*", "~gun", "~guns", "~gunsmith*", "~rifle*",
                            "~pistol*", "ammunition", "explosive*", "arms manufactur*",
                            "arms trad*", "arms deal*", "munition*", "armament*",
                            "military equipment",
                            "military hardware", "defense contract*", "defence contract*",
                            "defense industr*", "defence industr*", "armas", "armas de fuego",
                            "armamento*", "municion*", "explosivos", "industria de defensa",
                            "equipo militar", "material belico"),
    "adult content": ("adult content", "adult entertainment", "pornograph*",
                      "contenido para adultos", "entretenimiento para adultos"),
    "unlicensed gambling": ("gambling", "casino*", "betting", "bookmak*", "wager*",
                            "apuestas", "juegos de azar", "lottery", "loteria*"),
    "unlicensed virtual asset exchange": ("virtual asset exchange*", "crypto exchange*",
                                          "cryptocurrency exchange*",
                                          "exchange of virtual assets",
                                          "intercambio de activos virtuales",
                                          "digital asset exchange*", "crypto trading platform*",
                                          "casa de cambio de criptomonedas"),
    "shell banks": ("shell bank*", "banco pantalla", "bancos pantalla", "banco fantasma"),
    "cash intensive retail": ("cash intensive", "uso intensivo de efectivo",
                              "intensivo en efectivo", "convenience store*", "liquor store*",
                              "pawn*", "casa de empeno", "casas de empeno", "car wash*",
                              "tiendas de conveniencia"),
    # Packets describe trade finance by what it does more often than by name: an object
    # clause may speak of funding export cargoes and paying sellers against shipping
    # documents without ever using the term.
    "trade finance": ("trade finance", "trade financing", "financiamiento comercial",
                      "financiacion del comercio", "financiamiento al comercio",
                      "financiamiento de comercio exterior", f"export|import & {_FINANCING}",
                      f"{_FINANCING} & export*", f"{_FINANCING} & imports|importing|importer*",
                      f"{_FINANCING} & shipment*|embarque*",
                      f"{_FINANCING} & importacion*|importador*",
                      "prefinancing|prefinance|prefinanciamiento*|prefinanciacion* & export*",
                      "against bills of lading", "against bill of lading",
                      "contra conocimiento de embarque", "contra conocimientos de embarque",
                      "letters of credit", "letter of credit", "cartas de credito",
                      "carta de credito", "documentary credit*", "credito documentario",
                      "creditos documentarios", "documentary collection*",
                      "cobranza documentaria", f"commodity & {_FINANCING}"),
    "precious metals and stones": ("precious metal*", "precious stone*", "gemstone*",
                                   "diamond*", "bullion", "metales preciosos",
                                   "piedras preciosas", "diamantes", "lingotes", "~gold", "~oro",
                                   "~silver", "~plata", "jewel*", "joyeri*"),
    "real estate": ("real estate", "real property", "property development", "bienes inmuebles",
                    "bienes raices", "inmobiliari*", "inmuebles"),
    "money services": ("money service*", "money transmi*", "money transfer*", "remittance*",
                       "remesas", "currency exchange", "cambio de divisas", "casa de cambio",
                       "transferencia de dinero", "transferencias de dinero",
                       "money exchange", "check cashing", "payment service*",
                       "servicios de pago"),
    "virtual asset businesses licensed": ("virtual asset*", "digital asset*", "crypto*",
                                          "blockchain", "activos virtuales", "activos digitales",
                                          "criptoactivo*", "criptomoneda*"),
    "charities": ("charit*", "non profit", "nonprofit", "not for profit", "beneficencia",
                  "caridad", "sin fines de lucro", "sin animo de lucro", "organizacion benefica",
                  "fundraising", "recaudacion de fondos"),
    "unlicensed banking or money transmission": (
        "unlicensed bank*", "banca no autorizada", "captacion ilegal", "money transmi*",
        "transmision de dinero", "remesas no autorizadas",
    ),
    "shell bank": ("shell bank*", "banco pantalla", "bancos pantalla", "banco fantasma"),
    "adult entertainment": ("adult content", "adult entertainment", "pornograph*",
                            "contenido para adultos", "entretenimiento para adultos"),
    "online gambling": ("gambling", "online gambling", "casino*", "betting", "bookmak*", "wager*",
                        "apuestas", "juegos de azar", "lottery", "loteria*"),
    "weapons or defence brokerage": (
        "weapon*", "firearm*", "~gun", "~guns", "~gunsmith*", "~rifle*", "~pistol*",
        "ammunition", "explosive*", "arms manufactur*", "arms trad*", "arms deal*", "arms broker*",
        "munition*", "armament*", "military equipment", "military hardware",
        "defense contract*", "defence contract*", "defense broker*", "defence broker*",
        "defense industr*", "defence industr*", "armas", "armas de fuego", "armamento*",
        "municion*", "explosivos", "industria de defensa", "equipo militar", "material belico",
    ),
    "money service business": (
        "money service*", "money transmi*", "money transfer*", "remittance*", "remesas",
        "currency exchange", "cambio de divisas", "casa de cambio", "transferencia de dinero",
        "transferencias de dinero", "money exchange", "check cashing", "payment service*",
        "servicios de pago",
    ),
    "virtual asset or crypto exchange": (
        "virtual asset*", "digital asset*", "crypto*", "blockchain", "activos virtuales",
        "activos digitales", "criptoactivo*", "criptomoneda*", "intercambio de activos virtuales",
        "casa de cambio de criptomonedas",
    ),
    "precious metals or stones trading": (
        "precious metal*", "precious stone*", "gemstone*", "diamond*", "bullion",
        "metales preciosos", "piedras preciosas", "diamantes", "lingotes", "~gold", "~oro",
        "~silver", "~plata", "jewel*", "joyeri*",
    ),
    "maritime vessel chartering": (
        "vessel charter*", "ship charter*", "maritime charter*", "bareboat charter*",
        "time charter*", "fletamento maritimo", "fletamento de buques", "arrendamiento de naves",
        "naviera", "transporte maritimo de carga",
    ),
    "charitable foundation administration": (
        "charit*", "non profit", "nonprofit", "not for profit", "beneficencia", "caridad",
        "sin fines de lucro", "sin animo de lucro", "organizacion benefica", "fundraising",
        "recaudacion de fondos", "administracion de fundaciones", "philanthrop*",
    ),
}  # fmt: skip


def key(label: str) -> str:
    return fold(label)


def needs_licence(label: str) -> bool:
    return key(label).startswith("unlicensed ")


# Cues that the category is negated, excluded or only imitated ("toy pistols"). Folded text.
_HEDGES = (*NEGATIONS, "toy", "toys", "juguete*", "replica*", "imitation*", "airsoft", "paintball",
           "water gun*", "video game*", "videojuego*")  # fmt: skip
# Idioms that contain a cue word without negating anything: "sin fines de lucro" names a charity.
_IDIOMS = ("sin fines de lucro", "sin animo de lucro", "not for profit", "non profit",
           "no lucrativa", "no lucrativo")  # fmt: skip


def classify(label: str, activities: Iterable[str]) -> tuple[list[str], list[str]]:
    """(activities that name this category, activities that only might: a weak term alone, or a
    term under a negation, exclusion or imitation cue). `label` must be in LEXICON."""
    terms = LEXICON[key(label)]
    strong = [t for t in terms if not t.startswith("~")]
    weak = [t[1:] for t in terms if t.startswith("~")]
    named, hedged = [], []
    for activity in activities:
        text = fold(activity)
        if any(matches(t, text) for t in strong):
            bare = f" {text} "
            for idiom in _IDIOMS:
                bare = bare.replace(f" {fold(idiom)} ", " ")
            bare = bare.strip()
            negated = any(matches(n, bare) for n in _HEDGES)
            (hedged if negated else named).append(activity)
        elif any(matches(t, text) for t in weak):
            hedged.append(activity)
    return named, hedged
