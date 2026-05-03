import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import ee
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

from indice_espectral import calculate_indices

load_dotenv()

# ── Terminal formatação ──────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_VERDE = "\033[38;5;83m"
_VERMELHO = "\033[38;5;203m"
_CIANO = "\033[38;5;81m"
_AMARELO = "\033[38;5;221m"
_MAGENTA = "\033[38;5;213m"
_CINZA = "\033[38;5;245m"


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def _print(cor, simbolo, msg):
    print(f" {_CINZA}{_ts()}{_RESET} {cor}{simbolo}{_RESET} {msg}")


def info(msg):
    _print(_CIANO, "●", msg)


def ok(msg):
    _print(_VERDE, "✔", msg)


def erro(msg):
    _print(_VERMELHO, "✘", msg)


def aviso(msg):
    _print(_AMARELO, "⚠", msg)


def titulo(msg):
    print(f"\n  {_BOLD}{_CIANO}{'─'*60}{_RESET}")
    print(f"  {_BOLD}{_CIANO}   {msg}{_RESET}")
    print(f"  {_BOLD}{_CIANO}{'─'*60}{_RESET}")


def progresso(atual, total, data="", largura=20):
    if total == 0:
        return
    pct = atual / total
    n_preen = int(largura * pct)
    barra = "█" * n_preen + "░" * (largura - n_preen)
    cor_barra = _VERDE if pct >= 0.66 else _AMARELO if pct >= 0.33 else _VERMELHO
    sys.stdout.write(
        f"\r  {_CINZA}[{_RESET}{cor_barra}{barra}{_RESET}{_CINZA}]{_RESET}"
        f"  {_BOLD}{atual}{_RESET}/{total}"
        f"  {_CINZA}{data}{_RESET}"
    )
    sys.stdout.flush()


BANDAS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

ORDEM_COLUNAS = [
    "area",
    "data",
    "n_pixels",
    *[f"{b}_{stat}" for b in BANDAS for stat in ["mean", "median", "std"]],
    "NDVI",
    "NDMI",
    "EVI",
    "SAVI",
    "NBR",
]


def autenticar_ee():
    json_key = os.getenv("EARTH_ENGINE_KEY")
    if not json_key:
        raise ValueError("EARTH_ENGINE_KEY não encontrada.")
    info = json.loads(json_key)
    credenciais = ee.ServiceAccountCredentials(info["client_email"], key_data=json_key)
    ee.Initialize(credenciais)


def mascarar_nuvens(imagem):
    qa = imagem.select("QA60")
    nuvem_bit, cirrus_bit = 1 << 10, 1 << 11
    mascara = qa.bitwiseAnd(nuvem_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return imagem.updateMask(mascara).divide(10000)


def validar_arquivo(nome_arquivo):
    padrao = r"^(\d+)-(\d{8})\.shp$"
    match = re.match(padrao, nome_arquivo)
    if not match:
        return None, None
    cod_area, data_str = match.groups()
    data_dt = datetime.strptime(data_str, "%Y%m%d")
    return (cod_area, data_dt) if data_dt >= datetime(2022, 1, 1) else (None, None)


def processar_repositorio(pasta_saida: str):
    autenticar_ee()

    inicio = time.time()
    hoje = datetime.now()
    pasta_raiz = os.getenv("PASTA_AREAS", "areas")

    os.makedirs(pasta_saida, exist_ok=True)

    # Collect all areas first for total count
    areas = []
    for root, _, files in os.walk(pasta_raiz):
        for nome_arq in files:
            if nome_arq.endswith(".shp"):
                cod_area, data_cursor = validar_arquivo(nome_arq)
                if cod_area:
                    areas.append((root, nome_arq, cod_area, data_cursor))

    titulo("SENTINEL-2 — EXTRAÇÃO DE DADOS ORBITAIS")
    info(f"{len(areas)} área(s) encontrada(s) em '{pasta_raiz}'")
    print()

    for idx, (root, nome_arq, cod_area, data_cursor) in enumerate(areas, start=1):
        info(f"[{idx}/{len(areas)}] Processando área {_BOLD}{_VERDE}{cod_area}{_RESET}")

        try:
            gdf = gpd.read_file(os.path.join(root, nome_arq)).to_crs("EPSG:4326")
            geom_ee = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)

            colecao = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(geom_ee)
                .filterDate(
                    data_cursor.strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d")
                )
            )

            sys.stdout.write(f"  {_CINZA}⏳ Buscando imagens...{_RESET}")
            sys.stdout.flush()
            datas = colecao.aggregate_array("system:time_start").getInfo()
            datas = [datetime.utcfromtimestamp(ts / 1000) for ts in datas]
            sys.stdout.write("\r" + " " * 60 + "\r")
            info(f"{len(datas)} imagem(ns) disponível(is) para processamento")

            registros = []
            n_datas = len(datas)
            for i, data_img in enumerate(datas, start=1):
                s_data = data_img.strftime("%Y-%m-%d")

                progresso(i, n_datas, s_data)

                imagem = (
                    colecao.filterDate(
                        s_data,
                        (data_img + timedelta(days=1)).strftime("%Y-%m-%d"),
                    )
                    .map(mascarar_nuvens)
                    .select(BANDAS)
                    .median()
                )

                reducer = (
                    ee.Reducer.mean()
                    .combine(ee.Reducer.median(), "", True)
                    .combine(ee.Reducer.stdDev(), "", True)
                    .combine(ee.Reducer.count(), "", True)
                )

                stats = imagem.reduceRegion(
                    reducer=reducer, geometry=geom_ee, scale=10
                ).getInfo()

                if stats and any(v is not None for v in stats.values()):
                    stats_formatado = {}
                    n_pixels = None
                    for k, v in stats.items():
                        if k.endswith("_stdDev"):
                            stats_formatado[k.replace("_stdDev", "_std")] = v
                        elif k.endswith("_count"):
                            n_pixels = v
                        else:
                            stats_formatado[k] = v

                    registro = {
                        "area": cod_area,
                        "data": s_data,
                        "n_pixels": n_pixels,
                        **stats_formatado,
                    }
                    indices = calculate_indices(registro)
                    registro.update(indices)
                    registros.append(registro)

            sys.stdout.write("\n")

            if registros:
                caminho_json = os.path.join(pasta_saida, f"{cod_area}.json")
                try:
                    with open(caminho_json, "r", encoding="utf-8") as f:
                        dados_existentes = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    dados_existentes = []

                df_novo = pd.DataFrame(registros).reindex(columns=ORDEM_COLUNAS)
                df_atual = (
                    pd.DataFrame(dados_existentes) if dados_existentes else pd.DataFrame()
                )
                df_final = pd.concat([df_atual, df_novo], ignore_index=True)
                df_final = df_final.drop_duplicates(subset=["area", "data"])
                df_final.to_json(
                    caminho_json, orient="records", indent=2, force_ascii=False
                )

                ok(f"Salvo: {len(registros)} registro(s) em {os.path.basename(caminho_json)}")
            else:
                aviso(f"Nenhum dado obtido para área {cod_area}")

        except Exception as e:
            erro(f"{nome_arq}: {e}")

        print()

    duracao = time.time() - inicio
    titulo("PROCESSAMENTO CONCLUÍDO")
    ok(f"{len(areas)} área(s) processada(s) em {duracao:.1f}s")


def converter_para_xlsx(pasta_entrada: str, arquivo_saida: str):
    padrao_json = os.path.join(pasta_entrada, "*.json")
    arquivos = sorted(glob.glob(padrao_json))

    if not arquivos:
        aviso(f"Nenhum arquivo JSON encontrado em '{pasta_entrada}'.")
        return

    titulo("CONVERSÃO PARA XLSX")
    info(f"{len(arquivos)} arquivo(s) encontrado(s) em '{pasta_entrada}'")
    print()

    with pd.ExcelWriter(arquivo_saida, engine="openpyxl") as writer:
        for caminho_json in arquivos:
            nome_aba = os.path.splitext(os.path.basename(caminho_json))[0]
            with open(caminho_json, "r", encoding="utf-8") as f:
                dados = json.load(f)
            if not dados:
                aviso(f"{nome_aba}.json está vazio, ignorando.")
                continue
            df = pd.DataFrame(dados).reindex(columns=ORDEM_COLUNAS)
            df.to_excel(writer, sheet_name=nome_aba, index=False)
            ok(f"Aba '{nome_aba}' adicionada com {len(df)} registro(s).")

    ok(f"Planilha gerada: {arquivo_saida}")


def main():
    parser = argparse.ArgumentParser(
        description="Extração de dados orbitais Sentinel-2"
    )
    parser.add_argument(
        "--to-xlsx",
        metavar="DIR",
        nargs="?",
        const="__use_default__",
        help="Converter JSONs em DIR para XLSX (usa PASTA_SAIDA se omitido)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Arquivo XLSX de saída (apenas com --to-xlsx)",
    )

    args = parser.parse_args()

    if args.to_xlsx is not None:
        pasta_entrada = (
            args.to_xlsx
            if args.to_xlsx != "__use_default__"
            else os.getenv("PASTA_SAIDA", "analise")
        )
        arquivo_saida = args.output or os.getenv("NOME_EXCEL")
        if not arquivo_saida:
            nome_dir = os.path.basename(os.path.normpath(pasta_entrada))
            arquivo_saida = f"{nome_dir}.xlsx"
        converter_para_xlsx(pasta_entrada, arquivo_saida)
    else:
        pasta_saida = os.getenv("PASTA_SAIDA", "analise")
        processar_repositorio(pasta_saida)


if __name__ == "__main__":
    main()
