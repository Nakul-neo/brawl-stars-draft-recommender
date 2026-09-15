"""
Brawl Stars Draft Recommender & Win Predictor — deployable Gradio app.

This file does NOT train anything. It loads the pre-trained bundle
(model_bundle.joblib) produced once by the v3 Kaggle notebook's
"Export the deployable bundle" cell, and just serves the recommender UI.

Files needed alongside this one in the Space:
  - app.py                (this file)
  - requirements.txt
  - model_bundle.joblib   (exported from the notebook — see instructions)
"""

import os
import random
import itertools
import numpy as np
import joblib
import gradio as gr

# ---------------------------------------------------------------------------
# 1. Load the trained model + lookup tables (produced once in the notebook)
# ---------------------------------------------------------------------------
bundle = joblib.load("model_bundle.joblib")
clf = bundle["clf"]
STATS = bundle["STATS"]
E = bundle["E"]

all_brawlers = E["all_brawlers"]
all_maps = E["all_maps"]
all_modes = E["all_modes"]
BRAWLER_CLASS = E["BRAWLER_CLASS"]

# ---------------------------------------------------------------------------
# 2. Feature building — copied verbatim from the training notebook so a
#    prediction here is built exactly the same way it was during training.
# ---------------------------------------------------------------------------
def _mmm(vals):
    """mean / min / max over a short list -- plain Python, much faster than numpy here."""
    s = 0.0
    mn = 1e30
    mx = -1e30
    for v in vals:
        s += v
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    return s / len(vals), mn, mx


def build_feature_vector(S, E, my_entries, opp_entries, map_name, mode_name, time_frac=1.0):
    MP, MT = S["MEAN_POWER"], S["MEAN_TROPHIES"]
    my, my_pw, my_tr = [], [], []
    for e in my_entries:
        if isinstance(e, tuple):
            my.append(e[0]); my_pw.append(e[1]); my_tr.append(e[2])
        else:
            my.append(e); my_pw.append(MP); my_tr.append(MT)
    opp, opp_pw, opp_tr = [], [], []
    for e in opp_entries:
        if isinstance(e, tuple):
            opp.append(e[0]); opp_pw.append(e[1]); opp_tr.append(e[2])
        else:
            opp.append(e); opp_pw.append(MP); opp_tr.append(MT)

    n_b_, n_m_, n_mo_, n_r_ = E["n_b"], E["n_m"], E["n_mo"], E["n_roles"]
    b2i_, role2i_, cls_ = E["b2i"], E["role2i"], E["BRAWLER_CLASS"]
    row = np.zeros(E["FEATURE_LEN"], dtype=np.float32)

    role_base = 2 * n_b_ + n_m_ + n_mo_
    for b in my:
        row[b2i_[b]] = 1
        r = cls_.get(b)
        if r is not None:
            row[role_base + role2i_[r]] += 1
    for b in opp:
        row[n_b_ + b2i_[b]] = 1
        r = cls_.get(b)
        if r is not None:
            row[role_base + n_r_ + role2i_[r]] += 1

    map_name = map_name if map_name in E["m2i"] else "Unknown"
    if map_name in E["m2i"]:
        row[2 * n_b_ + E["m2i"][map_name]] = 1
    row[2 * n_b_ + n_m_ + E["mo2i"][mode_name]] = 1

    sh_o, sh_m, sh_d = S["sh_overall"], S["sh_map"], S["sh_mode"]
    sh_u, sh_y = S["sh_matchup"], S["sh_synergy"]
    rs_u, rs_y = S["res_matchup"], S["res_synergy"]

    if my:
        v = [sh_o.get(b, 0.5) for b in my]; my_ov = sum(v) / len(v); my_ov_min = min(v)
        v = [sh_m.get((b, map_name), 0.5) for b in my]; my_mp = sum(v) / len(v)
        v = [sh_d.get((b, mode_name), 0.5) for b in my]; my_md = sum(v) / len(v)
    else:
        my_ov = my_ov_min = my_mp = my_md = 0.5
    if opp:
        v = [sh_o.get(b, 0.5) for b in opp]; opp_ov = sum(v) / len(v); opp_ov_min = min(v)
        v = [sh_m.get((b, map_name), 0.5) for b in opp]; opp_mp = sum(v) / len(v)
        v = [sh_d.get((b, mode_name), 0.5) for b in opp]; opp_md = sum(v) / len(v)
    else:
        opp_ov = opp_ov_min = opp_mp = opp_md = 0.5

    if my and opp:
        c_mean, c_min, c_max = _mmm([sh_u.get((b, o), 0.5) for b in my for o in opp])
        cr_mean, cr_min, cr_max = _mmm([rs_u.get((b, o), 0.0) for b in my for o in opp])
    else:
        c_mean = c_min = c_max = 0.5
        cr_mean = cr_min = cr_max = 0.0

    def syn(bs):
        if len(bs) < 2:
            return (0.5, 0.5, 0.5, 0.0, 0.0, 0.0)
        pairs = list(itertools.combinations(bs, 2))
        return _mmm([sh_y.get(frozenset(p), 0.5) for p in pairs]) + _mmm(
            [rs_y.get(frozenset(p), 0.0) for p in pairs]
        )

    ms, os_ = syn(my), syn(opp)

    if not my_pw: my_pw = [MP]
    if not opp_pw: opp_pw = [MP]
    if not my_tr: my_tr = [MT]
    if not opp_tr: opp_tr = [MT]
    my_power = sum(my_pw) / len(my_pw); opp_power = sum(opp_pw) / len(opp_pw)
    my_troph = sum(my_tr) / len(my_tr); opp_troph = sum(opp_tr) / len(opp_tr)

    row[role_base + 2 * n_r_ :] = (
        len(my), len(opp),
        my_ov, opp_ov, my_ov_min, opp_ov_min, my_ov - opp_ov,
        my_mp, opp_mp, my_mp - opp_mp, my_md, opp_md, my_md - opp_md,
        c_mean, c_min, c_max, cr_mean, cr_min, cr_max,
        ms[0], ms[1], ms[2], ms[3], ms[4], ms[5],
        os_[0], os_[1], os_[2], os_[3], os_[4], os_[5],
        my_power, opp_power, my_power - opp_power, min(my_pw), min(opp_pw),
        my_troph, opp_troph, my_troph - opp_troph,
        time_frac,
    )
    return row


# ---------------------------------------------------------------------------
# 3. Recommender + win predictor (same logic as the notebook)
# ---------------------------------------------------------------------------
def recommend_picks(my_picks, opp_picks, map_name, mode_name, banned=None, owned=None, top_k=3):
    banned = set(banned or [])
    my_names = [e[0] if isinstance(e, tuple) else e for e in my_picks]
    opp_names = [e[0] if isinstance(e, tuple) else e for e in opp_picks]
    taken = set(my_names) | set(opp_names) | banned
    pool = set(owned) if owned is not None else set(all_brawlers)
    candidates = [b for b in all_brawlers if b in pool and b not in taken]
    if not candidates:
        return []
    feats = np.array(
        [build_feature_vector(STATS, E, my_picks + [c], opp_picks, map_name, mode_name) for c in candidates]
    )
    p = clf.predict_proba(feats)[:, 1]
    ranked = sorted(zip(candidates, p), key=lambda x: -x[1])
    return [(b, pr, BRAWLER_CLASS.get(b, "Unknown")) for b, pr in ranked[:top_k]]


def calculate_win_probability(teamA, teamB, map_name, mode_name):
    fA = build_feature_vector(STATS, E, teamA, teamB, map_name, mode_name)
    fB = build_feature_vector(STATS, E, teamB, teamA, map_name, mode_name)
    pA = clf.predict_proba(fA.reshape(1, -1))[0, 1]
    pB = clf.predict_proba(fB.reshape(1, -1))[0, 1]
    winA = (pA + (1 - pB)) / 2
    return float(winA), float(1 - winA)


# ---------------------------------------------------------------------------
# 4. Gradio UI — turn-by-turn draft with live recommendations
# ---------------------------------------------------------------------------
def display_teams(s):
    return f"**RED:** {', '.join(s['red']) or '-'}\n\n**BLUE:** {', '.join(s['blue']) or '-'}"


def get_current_recommendation(s):
    i = s["pick_idx"]
    if i >= 6:
        return "Draft complete.", []
    side = s["seq"][i]
    my, opp = (s["red"], s["blue"]) if side == "RED" else (s["blue"], s["red"])
    owned = s["red_owned"] if side == "RED" else s["blue_owned"]
    rec = recommend_picks(my, opp, s["map"], s["mode"], owned=owned, top_k=5)
    if not rec:
        return f"No brawlers left for **{side}**.", []
    lines = [f"### Turn {i + 1} - {side} to pick"]
    choices = []
    for b, p, c in rec:
        lab = f"{b} ({c}) - {p:.1%}"
        choices.append(lab)
        lines.append(f"- {lab}")
    taken = set(my) | set(opp)
    pool = owned if owned is not None else set(all_brawlers)
    choices += sorted([b for b in pool if b not in taken and b not in [r[0] for r in rec]])
    return "\n".join(lines), choices


def start_draft(mp, md, ro, bo):
    toss = random.choice(["RED", "BLUE"])
    other = "BLUE" if toss == "RED" else "RED"
    s = {
        "red": [], "blue": [], "pick_idx": 0, "map": mp, "mode": md,
        "red_owned": set(ro), "blue_owned": set(bo),
        "seq": [toss, other, other, toss, toss, other],
    }
    info, ch = get_current_recommendation(s)
    return s, f"Coin toss: **{toss}** picks first!\n\n{info}", gr.update(choices=ch, value=None), display_teams(s)


def make_pick(s, sel):
    if not sel or s["pick_idx"] >= 6:
        return s, "Pick a brawler first.", gr.update(), display_teams(s)
    b = sel.split(" (")[0]
    side = s["seq"][s["pick_idx"]]
    (s["red"] if side == "RED" else s["blue"]).append(b)
    s["pick_idx"] += 1
    if s["pick_idx"] >= 6:
        wR, wB = calculate_win_probability(s["red"], s["blue"], s["map"], s["mode"])
        final = (
            f"### Draft complete!\n\n**RED:** {', '.join(s['red'])}\n\n**BLUE:** {', '.join(s['blue'])}\n\n"
            f"### Win probability -> RED: {wR:.1%} | BLUE: {wB:.1%}"
        )
        return s, final, gr.update(choices=[], value=None), display_teams(s)
    info, ch = get_current_recommendation(s)
    return s, info, gr.update(choices=ch, value=None), display_teams(s)


with gr.Blocks(title="Brawl Stars Draft Recommender") as demo:
    gr.Markdown(
        "# Brawl Stars Draft Recommender & Win Predictor\n"
        "Pick a map and mode, flip the coin, and get a live pick recommendation for every turn "
        "of a 1-2-2-1 draft, plus a final win-probability call.\n\n"
        "*Trained on ~118k ranked matches. Full-draft AUC ≈ 0.70 on a held-out test set — "
        "this is a directional signal, not a guarantee.*"
    )
    state = gr.State(
        {"red": [], "blue": [], "pick_idx": 0, "map": None, "mode": None,
         "red_owned": None, "blue_owned": None, "seq": None}
    )
    with gr.Row():
        map_dd = gr.Dropdown(choices=all_maps, value=all_maps[0], label="Map")
        mode_dd = gr.Dropdown(choices=all_modes, value=all_modes[0], label="Mode")
    with gr.Accordion("Restrict to owned brawlers (optional)", open=False):
        with gr.Row():
            red_owned = gr.CheckboxGroup(choices=all_brawlers, value=all_brawlers, label="RED owns")
            blue_owned = gr.CheckboxGroup(choices=all_brawlers, value=all_brawlers, label="BLUE owns")
    start_btn = gr.Button("Flip coin & start draft", variant="primary")
    turn_md = gr.Markdown()
    pick_dd = gr.Dropdown(choices=[], label="Choose a pick")
    pick_btn = gr.Button("Confirm pick")
    teams_md = gr.Markdown()

    start_btn.click(start_draft, [map_dd, mode_dd, red_owned, blue_owned], [state, turn_md, pick_dd, teams_md])
    pick_btn.click(make_pick, [state, pick_dd], [state, turn_md, pick_dd, teams_md])

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
