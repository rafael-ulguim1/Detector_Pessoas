# Utilitários do pacote
import argparse

"""
    parse_args
    Função para parsear os argumentos da linha de comando  
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Meu pacote python xyz")

    parser.add_argument(
        "-p",
        "--problema",
        required=True,
        help="Problema exemplo a ser resolvido",
    )

    return parser.parse_args()
