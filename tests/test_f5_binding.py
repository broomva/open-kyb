from kyb.witness.bind import Scope, binding


def rr_tuple(subject, attr, value, struck=False):
    return {"subject": subject, "attribute": attr, "value": value, "struck": struck}


def test_fix1_entity_name_as_subject():
    # c-B01-03 positive
    claim = {
        "subject": "IBERIA SOFTWARE S.L.",
        "type": "legal_name",
        "value": "IBERIA SOFTWARE S.L.",
    }
    rr = {
        "values": [
            rr_tuple("LUCAS NAVARRO SOLÍS", "nationality", "mexicano"),
            rr_tuple(
                "IBERIA SOFTWARE S.L.",
                "type",
                "Sociedad Anonima de Capital Variado",
            ),
        ]
    }
    transcription = "IBERIA SOFTWARE S.L."
    assert binding(claim, rr, transcription)

    # negative: company name not in crop
    claim_wrong = {
        "subject": "OTRA EMPRESA, S.A.",
        "type": "legal_name",
        "value": "OTRA EMPRESA, S.A.",
    }
    assert not binding(claim_wrong, rr, "OTRA EMPRESA, S.A.")


def test_fix2_field_label_subjects():
    # c-B01-01 positive
    claim = {
        "subject": "IBERIA SOFTWARE S.L.",
        "type": "deed_number",
        "value": "48.223",
    }
    rr = {
        "values": [
            rr_tuple("ESCRITURA PÚBLICA", "numero", "48.223"),
        ]
    }

    # The document scope has GRUPO IBERIA as entity
    class Fact:
        def __init__(self, subject, attr, value):
            self.subject = subject
            self.attribute = attr
            self.value = value
            self.block_id = 1

    scope = Scope(
        ["ACTA CONSTITUTIVA"],
        [
            Fact(
                "IBERIA SOFTWARE S.L.",
                "legal_name",
                "IBERIA SOFTWARE S.L.",
            )
        ],
        {"1": rr},
    )
    claim["block_id"] = 1
    assert binding(claim, rr, scope=scope)

    # c-B01-02 positive
    claim2 = {
        "subject": "IBERIA SOFTWARE S.L.",
        "type": "incorporation_date",
        "value": "2020-04-12",
        "block_id": 1,
    }
    rr2 = {
        "values": [
            rr_tuple("ESCRITURA PÚBLICA", "date", "2020-04-12"),
            rr_tuple("Notaría Pública", "numero", "14"),
        ]
    }
    assert binding(claim2, rr2, scope=scope)

    # negative: right value with wrong subject
    claim3 = {
        "subject": "OTRO SUJETO",
        "type": "incorporation_date",
        "value": "2020-04-12",
        "block_id": 1,
    }
    assert not binding(claim3, rr2, scope=scope)

    # negative: wrong value with right subject
    claim4 = {
        "subject": "IBERIA SOFTWARE S.L.",
        "type": "incorporation_date",
        "value": "2018-01-01",
        "block_id": 1,
    }
    assert not binding(claim4, rr2, scope=scope)

    # negative: a label subject while another named party holds the attribute
    rr3 = {
        "values": [
            rr_tuple("ESCRITURA PÚBLICA", "numero", "48.223"),
            rr_tuple("OTRO SUJETO", "deed_number", "99.999"),
        ]
    }
    assert not binding(claim, rr3, scope=scope)


def test_fix3_punctuation_and_role_expansion():
    # c-B05-03 positive (role expansion)
    claim = {"subject": "Lucas Navarro Solís", "type": "role", "value": "Presidente"}
    rr = {
        "values": [
            rr_tuple(
                "Lucas Navarro Solís",
                "Asistentes",
                "Presidente del Consejo de Administración",
            )
        ]
    }
    assert not binding(claim, rr)

    # negative: role prefix with a different person
    claim2 = {"subject": "Valeria Ortega", "type": "role", "value": "Presidente"}
    assert not binding(claim2, rr)

    # punctuation stripping cannot merge different values
    # "5.000" vs "6.000.000"
    claim_num = {
        "subject": "Lucas Navarro Solís",
        "attribute": "shares",
        "type": "identifier",
        "value": "5.000",
    }
    rr_num = {"values": [rr_tuple("Lucas Navarro Solís", "cantidad", "6.000.000")]}
    assert not binding(claim_num, rr_num)

    # "ACTIVO" vs "INACTIVO"
    claim_stat = {
        "subject": "Lucas Navarro Solís",
        "attribute": "tax_status",
        "type": "identifier",
        "value": "ACTIVO",
    }
    rr_stat = {"values": [rr_tuple("Lucas Navarro Solís", "estatus", "INACTIVO")]}
    assert not binding(claim_stat, rr_stat)

    # ACTIVO. vs ACTIVO
    claim_stat2 = {
        "subject": "Lucas Navarro Solís",
        "attribute": "tax_status",
        "type": "identifier",
        "value": "ACTIVO.",
    }
    rr_stat2 = {"values": [rr_tuple("Lucas Navarro Solís", "estatus", "ACTIVO")]}
    assert binding(claim_stat2, rr_stat2)


def test_p1a_legal_name_rival():
    claim_p1a = {
        "subject": "INVERSIONES ALTIPLANO, S.A.",
        "type": "legal_name",
        "value": "IBERIA SOFTWARE S.L.",
    }
    rr = {
        "values": [
            rr_tuple("IBERIA SOFTWARE S.L.", "type", "Sociedad"),
        ]
    }
    assert not binding(claim_p1a, rr, "IBERIA SOFTWARE S.L.")


def test_p1b_role_no_prefix():
    claim_p1b = {"subject": "Juan", "type": "role", "value": "administrador"}
    rr = {"values": [rr_tuple("Juan", "cargo", "administrador suplente")]}
    assert not binding(claim_p1b, rr)


def test_p1c_no_notaria_or_address_on_doc():
    claim = {
        "subject": "EMPRESA",
        "type": "residential_address",
        "value": "Calle 123",
        "block_id": 1,
    }
    rr = {
        "values": [
            rr_tuple("notaria", "address", "Calle 123"),
        ]
    }
    scope = Scope(["ACTA"], [], {"1": rr})
    assert not binding(claim, rr, scope=scope)
