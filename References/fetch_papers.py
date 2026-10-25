#!/usr/bin/env python3
"""
fetch_papers.py — download the open-access PDFs for the RL / adaptive cancer
therapy reading list.

Usage:
    pip install requests
    python fetch_papers.py --email you@smail.iitm.ac.in

Resolution order for each entry:
    1. hard-coded preprint URL (arXiv / bioRxiv / medRxiv / JMLR / publisher OA)
    2. Unpaywall DOI lookup  -> best legal OA copy
    3. report as "needs institutional access" and emit the DOI link

Nothing here touches paywalls or shadow libraries. Anything it can't get
legally is listed at the end so you can pull it through the IITM library
proxy (https://library.iitm.ac.in -> Remote Access / e-resources).
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse

try:
    import requests
except ImportError:
    sys.exit("Need requests:  pip install requests")


OUTDIR = "papers"
UA = "Mozilla/5.0 (compatible; reading-list-fetcher/1.0)"
PAUSE = 1.5  # be polite to the servers


def arx(i):
    return [f"https://arxiv.org/pdf/{i}"]


def bio(d):
    """bioRxiv/medRxiv: try a few versions, newest-looking first."""
    host = "www.medrxiv.org" if d.startswith("10.1101/2024.04.08.243") else "www.biorxiv.org"
    return [f"https://{host}/content/{d}v{v}.full.pdf" for v in (3, 2, 1)]


# key, title, doi, [candidate direct PDF urls], section
PAPERS = [
    # --- 1. Evolutionary substrate -------------------------------------
    ("01_gatenby2009_adaptive_therapy",
     "Adaptive Therapy",
     "10.1158/0008-5472.CAN-08-3658", [], "1-foundations"),

    ("02_enriquez-navas2016_breast_preclinical",
     "Exploiting evolutionary principles to prolong tumor control in preclinical models of breast cancer",
     "10.1126/scitranslmed.aad7842", [], "1-foundations"),

    ("03_zhang2017_mcrpc_abiraterone",
     "Integrating evolutionary dynamics into treatment of metastatic castrate-resistant prostate cancer",
     "10.1038/s41467-017-01968-5", [], "1-foundations"),

    ("04_zhang2022_elife_abiraterone",
     "Evolution-based mathematical models significantly prolong response to abiraterone in mCRPC",
     "10.7554/eLife.76284",
     ["https://elifesciences.org/articles/76284.pdf"], "1-foundations"),

    ("05a_mistry2021_critique",
     "On the reporting and analysis of a cancer evolutionary adaptive dosing trial",
     "10.1038/s41467-020-20174-4", [], "1-foundations"),

    ("05b_zhang2021_response_to_mistry",
     "Response to Mistry",
     "10.1038/s41467-020-20175-3", [], "1-foundations"),

    ("06_viossat2021_tumour_containment",
     "A theoretical analysis of tumour containment",
     "10.1038/s41559-021-01428-w", [], "1-foundations"),

    ("07_strobl2021_turnover_cost_of_resistance",
     "Turnover modulates the need for a cost of resistance in adaptive therapy",
     "10.1158/0008-5472.CAN-20-0806", [], "1-foundations"),

    ("08_kaznatcheev2019_fibroblasts_alectinib",
     "Fibroblasts and alectinib switch the evolutionary games played by non-small cell lung cancer",
     "10.1038/s41559-018-0768-z",
     ["https://www.biorxiv.org/content/10.1101/179259v6.full.pdf"], "1-foundations"),

    ("09_west2023_open_questions",
     "A survey of open questions in adaptive therapy",
     "10.7554/eLife.84263",
     arx("2210.12062") + ["https://elifesciences.org/articles/84263.pdf"], "1-foundations"),

    # --- 2. Optimal control / dynamic programming -----------------------
    ("10_gluzman2020_dp_egt",
     "Optimizing adaptive cancer therapy: dynamic programming and evolutionary game theory",
     "10.1098/rspb.2019.2454", arx("1812.01805"), "2-optimal-control"),

    ("11_wang2024_threshold_awareness",
     "Threshold-awareness in adaptive cancer therapy",
     "10.1371/journal.pcbi.1012165",
     ["https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1012165&type=printable"],
     "2-optimal-control"),

    ("12a_cunningham2018_optimal_control_mcrpc",
     "Optimal control to develop therapeutic strategies for metastatic castrate resistant prostate cancer",
     "10.1016/j.jtbi.2018.09.022", [], "2-optimal-control"),

    ("12b_cunningham2020_eco_evo_stability",
     "Optimal control to reach eco-evolutionary stability in metastatic castrate-resistant prostate cancer",
     "10.1371/journal.pone.0243386",
     ["https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0243386&type=printable"],
     "2-optimal-control"),

    ("13_west2020_multidrug_adaptive",
     "Towards multidrug adaptive therapy",
     "10.1158/0008-5472.CAN-19-2669", [], "2-optimal-control"),

    ("14_npj2026_multidrug_optimal_control",
     "Optimal control theory as a method for designing multidrug adaptive therapy regimens",
     "10.1038/s41540-025-00613-y", [], "2-optimal-control"),

    # --- 3. Core RL for adaptive / evolutionary therapy -----------------
    ("15_engelhardt2020_celludose",
     "Dynamic Control of Stochastic Evolution: A DRL Approach to Adaptively Targeting Emergent Drug Resistance",
     None,
     ["https://jmlr.org/papers/volume21/19-546/19-546.pdf"] + arx("1903.11373"), "3-core-rl"),

    ("16_gallagher2023_learning_to_adapt_preprint",
     "Learning to Adapt - Deep Reinforcement Learning in Treatment-Resistant Prostate Cancer",
     "10.1101/2023.04.28.538766", bio("10.1101/2023.04.28.538766"), "3-core-rl"),

    ("17_gallagher2024_model_driven_dl",
     "Mathematical Model-Driven Deep Learning Enables Personalized Adaptive Therapy",
     "10.1158/0008-5472.CAN-23-2040", [], "3-core-rl"),

    ("18_weaver2024_rl_antibiotic_resistance",
     "Reinforcement learning informs optimal treatment strategies to limit antibiotic resistance",
     "10.1073/pnas.2303165121", bio("10.1101/2023.01.12.523765"), "3-core-rl"),

    ("19_king_fitness_seascapes",
     "Fitness seascapes are necessary for realistic modeling of the evolutionary response to drug therapy",
     "10.1126/sciadv.adv1268", [], "3-core-rl"),

    ("20_reinforcement_failing_2025",
     "Reinforcement Failing guides the discovery of emergent physical dynamics in adaptive tumor therapy",
     "10.1101/2025.04.08.647768", bio("10.1101/2025.04.08.647768"), "3-core-rl"),

    ("21_m4rl_glioblastoma_sciadv",
     "Multiscale mathematical model-informed RL optimizes combination treatment scheduling in glioblastoma",
     "10.1126/sciadv.adv3316", [], "3-core-rl"),

    ("22_single_threshold_partial_cycle",
     "Single-threshold-guided adaptive cancer therapy with partial-cycle treatment",
     None, [], "3-core-rl"),

    ("23_memory_driven_rl_prostate",
     "A memory-driven reinforcement learning model of phenotypic adaptation ... prostate cancer",
     "10.1101/2025.11.03.686133", bio("10.1101/2025.11.03.686133"), "3-core-rl"),

    ("24_stackelberg_breast_rl",
     "RL-Driven Evolutionary Stackelberg Game Model for Adaptive Breast Cancer Therapy",
     "10.3390/mca30060134",
     ["https://www.mdpi.com/2297-8747/30/6/134/pdf"], "3-core-rl"),

    ("25_physrev_e_qlearning_stochastic_tumor",
     "Reinforcement learning for chemotherapy scheduling in a stochastic tumor evolution model",
     "10.1103/cbkw-5s4b", [], "3-core-rl"),

    ("25b_optimal_adaptive_egt_plosone",
     "Optimal adaptive cancer therapy based on evolutionary game theory",
     "10.1371/journal.pone.0320677",
     ["https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0320677&type=printable"],
     "3-core-rl"),

    # --- 4. RL dosing under toxicity / uncertainty ----------------------
    ("26_padmanabhan2017_rl_drug_dosing",
     "Reinforcement learning-based control of drug dosing for cancer chemotherapy treatment",
     "10.1016/j.mbs.2017.08.004", [], "4-rl-dosing"),

    ("27_eastman2021_robust_schedules",
     "Reinforcement learning derived chemotherapeutic schedules for robust patient-specific therapy",
     "10.1038/s41598-021-97028-6", bio("10.1101/2021.04.23.441182"), "4-rl-dosing"),

    ("28_kiram2026_recurrent_drl_pomdp",
     "Recurrent Deep Reinforcement Learning for Chemotherapy Control under Partial Observability",
     None, arx("2605.02552"), "4-rl-dosing"),

    ("29_tordesillas2019_dqn_ddpg_vs_oc",
     "Personalized Cancer Chemotherapy Schedule: model-based vs model-free scheduling",
     None, arx("1904.01200"), "4-rl-dosing"),

    ("30_socr_offline_rl_chemo",
     "Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning",
     "10.1109/JBHI.2022.3183854", [], "4-rl-dosing"),

    ("30b_drl_chemo_drug_dose_control",
     "Deep reinforcement learning-based control of chemo-drug dose in cancer treatment",
     "10.1016/j.cmpb.2023.107884", [], "4-rl-dosing"),

    # --- 5. Adjacent modalities ----------------------------------------
    ("31_tseng2017_drl_radiation_adaptation",
     "Deep reinforcement learning for automated radiation adaptation in lung cancer",
     "10.1002/mp.12625", [], "5-adjacent"),

    ("31b_niraula_quantum_drl_radiotherapy",
     "Quantum deep reinforcement learning for clinical decision support in oncology: adaptive radiotherapy",
     "10.1038/s41598-021-02910-y", [], "5-adjacent"),

    ("32_ici_drl_time_modeling",
     "Optimized patient-specific immune checkpoint inhibitor therapies based on TIME modeling",
     "10.1093/bib/bbae547", bio("10.1101/2024.04.08.24305526"), "5-adjacent"),

    ("33_abm_csc_rl_radiotherapy",
     "A biologically constrained agent-based model of cancer stem cell dynamics with RL-guided adaptive radiotherapy",
     "10.1371/journal.pone.0340426",
     ["https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0340426&type=printable"],
     "5-adjacent"),

    ("33b_ddpg_tumor_immune",
     "Deep learning for adaptive chemotherapy: a DDPG-based approach to optimizing tumor-immune dynamics",
     "10.1371/journal.pone.0345877",
     ["https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0345877&type=printable"],
     "5-adjacent"),
]


def looks_like_pdf(resp):
    ctype = resp.headers.get("content-type", "").lower()
    return "pdf" in ctype or resp.content[:5] == b"%PDF-"


def try_download(session, url, path):
    try:
        r = session.get(url, timeout=45, allow_redirects=True)
    except requests.RequestException as e:
        return False, f"{type(e).__name__}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    if not looks_like_pdf(r):
        return False, "not a PDF (paywall/landing page)"
    if len(r.content) < 20_000:
        return False, f"suspiciously small ({len(r.content)} B)"
    with open(path, "wb") as f:
        f.write(r.content)
    return True, f"{len(r.content)//1024} KB"


def unpaywall_pdf(session, doi, email):
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={email}"
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return None
        loc = (r.json() or {}).get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True,
                    help="your email — required by the Unpaywall API")
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    got, missing, rows = [], [], []

    for key, title, doi, direct, section in PAPERS:
        path = os.path.join(args.outdir, key + ".pdf")
        if os.path.exists(path) and os.path.getsize(path) > 20_000:
            print(f"  [skip]  {key}")
            got.append(key)
            continue

        print(f"\n>> {key}\n   {title[:80]}")
        candidates = list(direct)
        if doi:
            up = unpaywall_pdf(session, doi, args.email)
            if up and up not in candidates:
                candidates.append(up)
            time.sleep(0.5)

        ok, why = False, "no OA source found"
        for url in candidates:
            ok, why = try_download(session, url, path)
            print(f"   {'OK ' if ok else '-- '} {url[:95]}  ({why})")
            if ok:
                break
            time.sleep(PAUSE)

        (got if ok else missing).append(key)
        rows.append({
            "key": key, "section": section, "title": title,
            "doi": doi or "", "downloaded": ok,
            "doi_link": f"https://doi.org/{doi}" if doi else "",
        })
        time.sleep(PAUSE)

    with open(os.path.join(args.outdir, "_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["key", "section", "title", "doi", "downloaded", "doi_link"])
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    print(f"Downloaded : {len(got)}")
    print(f"Not open   : {len(missing)}")
    if missing:
        print("\nPull these through the IITM library proxy — DOIs:")
        for key, title, doi, _, _ in PAPERS:
            if key in missing:
                link = f"https://doi.org/{doi}" if doi else "(search by title)"
                print(f"  {link}\n      {title[:88]}")
    print(f"\nManifest: {os.path.join(args.outdir, '_manifest.csv')}")


if __name__ == "__main__":
    main()
