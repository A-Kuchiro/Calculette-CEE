import streamlit as st
import yaml
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

# Dossier des YAML
PARAMS_DIR = Path(__file__).parent / "params"

# ------------------------------
# Chargement & helpers
# ------------------------------
def load_fiches():
    fiches = []
    for p in sorted(PARAMS_DIR.glob("*.yaml")):
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            data["_file"] = p.name
            fiches.append(data)
    return fiches

def D(x):
    return Decimal(str(x))

def arrondir(val, ndigits=0):
    q = Decimal("1").scaleb(-ndigits)
    return val.quantize(q, rounding=ROUND_HALF_UP)

def render_input(field):
    t = field.get("type")
    name = field["name"]
    label = field.get("label", name)
    if t == "number":
        # Récupération brute des bornes/valeurs/step sans imposer de type
        min_v = field.get("min", None)
        max_v = field.get("max", None)
        val_v = field.get("default", None)
        step_v = field.get("step", None)

        # Détermination du type cible (int vs float) pour numéroter de façon homogène
        # Règle: si l’un des paramètres est float -> tout en float, sinon tout en int
        nums = [v for v in (min_v, max_v, val_v, step_v) if v is not None]
        use_float = any(isinstance(v, float) for v in nums)

        if use_float:
            min_value = float(min_v) if min_v is not None else float(0.0)
            max_value = float(max_v) if max_v is not None else float(1e12)
            value = float(val_v) if val_v is not None else float(0.0)
            step = float(step_v) if step_v is not None else float(1.0)
        else:
            min_value = int(min_v) if min_v is not None else int(0)
            max_value = int(max_v) if max_v is not None else int(10**12)
            value = int(val_v) if val_v is not None else int(0)
            step = int(step_v) if step_v is not None else int(1)

        v = st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            key=f"inp_{name}",
        )
        return name, v

    elif t == "select":
        opts = field.get("options", [])
        labels = [o["label"] if isinstance(o, dict) else str(o) for o in opts]
        values = [o["value"] if isinstance(o, dict) else o for o in opts]
        if not labels:
            st.error(f"Aucune option pour {label}")
            st.stop()
        vlabel = st.selectbox(label, labels, index=0, key=f"inp_{name}")
        return name, values[labels.index(vlabel)]
    else:
        st.error(f"Type de champ non géré: {t}")
        st.stop()

def _guess_number(values, key):
    if key is None:
        return None
    v = values.get(key)
    try:
        return D(v)
    except Exception:
        return None

def _pick_variant(calc_spec, values):
    """Si calcul_standard contient des 'variants', retourne le block de la variante dont les conditions 'when' matchent."""
    variants = calc_spec.get("variants")
    if not variants:
        return calc_spec  # déjà un block simple
    for var in variants:
        cond = var.get("when", {})
        ok = True
        for k, expected in cond.items():
            if str(values.get(k)) != str(expected):
                ok = False
                break
        if ok:
            return var.get("block", {})
    st.error("Aucune variante ne correspond aux paramètres fournis.")
    st.stop()

def _choose_band(bands, x):
    """Trouve la bande où min ≤ x < max (si min/max non nuls)."""
    for b in bands:
        bmin = b.get("min")
        bmax = b.get("max")
        ok_min = True if bmin is None else (x >= D(bmin))
        ok_max = True if bmax is None else (x < D(bmax))
        if ok_min and ok_max:
            return b
    return None

# ------------------------------
# Moteur de calcul
# ------------------------------
def compute_block(block, values):
    kind = block.get("kind", "multiplication")
    out = block.get("output", {"name": "resultat", "round": 0})
    name = out.get("name", "resultat")
    rnd = out.get("round", 0)

    # --- (1) multiplication : base_input * (rate | rates[factor_input]) ---
    if kind == "multiplication":
        base_key = block.get("base_input")
        factor_key = block.get("factor_input")
        rates = block.get("rates", {})
        detail = ""

        base_val = _guess_number(values, base_key) if base_key else None
        if base_val is None:
            base_val = D(1)
            base_desc = "1 (forfait)"
        else:
            base_desc = f"{base_key}={base_val}"

        if factor_key:
            k = str(values.get(factor_key))
            if k not in rates:
                st.error(f"Taux indisponible pour {factor_key}={k}")
                st.stop()
            rate = D(rates[k])
            detail = f"{base_desc} × taux[{k}]={rate}"
        else:
            if "rate" not in block:
                st.error("Aucun taux défini (ni rates ni rate).")
                st.stop()
            rate = D(block["rate"])
            detail = f"{base_desc} × taux={rate}"

        raw = base_val * rate
        return {
            "name": name,
            "value": arrondir(raw, rnd),
            "raw": raw,
            "detail": f"{detail} → {raw}",
            "unit": None,
        }

    # --- (2) fixed_by_factor : montant forfaitaire choisi par factor_input ---
    if kind in ("fixed_by_factor", "forfait_par_facteur"):
        factor_key = block.get("factor_input")
        rates = block.get("rates", {})
        k = str(values.get(factor_key))
        if k not in rates:
            st.error(f"Montant forfaitaire indisponible pour {factor_key}={k}")
            st.stop()
        raw = D(rates[k])
        return {
            "name": name,
            "value": arrondir(raw, rnd),
            "raw": raw,
            "detail": f"forfait[{factor_key}={k}] = {raw}",
            "unit": None,
        }

    # --- (3) band_table : choisit une bande (min ≤ x < max), lit le taux par zone (forfait) ---
    if kind == "band_table":
        band_key = block.get("band_input")       # ex: etas
        factor_key = block.get("factor_input")   # ex: zone_climatique
        bands = block.get("bands", [])
        if not band_key or not bands or not factor_key:
            st.error("Configuration band_table incomplète (band_input / factor_input / bands).")
            st.stop()

        x = _guess_number(values, band_key)
        if x is None:
            st.error(f"Valeur numérique requise pour '{band_key}'.")
            st.stop()

        chosen = _choose_band(bands, x)
        if not chosen:
            st.error(f"Aucune bande trouvée pour {band_key}={x}.")
            st.stop()

        zone = str(values.get(factor_key))
        rates = chosen.get("rates", {})
        if zone not in rates:
            st.error(f"Taux indisponible pour zone={zone} dans la bande '{chosen.get('name','')}'.")
            st.stop()

        rate = D(rates[zone])
        raw = rate
        detail = f"bande '{chosen.get('name','')}' ({band_key}={x}) → taux[{zone}]={rate}"
        return {
            "name": name,
            "value": arrondir(raw, rnd),
            "raw": raw,
            "detail": detail,
            "unit": None,
        }

    # --- (3bis) band_table_times_base : comme band_table, puis × base_input (ex: puissance W) ---
    if kind == "band_table_times_base":
        band_key = block.get("band_input")       # ex: efficacite
        factor_key = block.get("factor_input")   # ex: secteur
        base_key = block.get("base_input")       # ex: puissance_w
        bands = block.get("bands", [])
        if not (band_key and factor_key and base_key and bands):
            st.error("Configuration band_table_times_base incomplète.")
            st.stop()

        x = _guess_number(values, band_key)
        base_val = _guess_number(values, base_key)
        if x is None or base_val is None:
            st.error(f"Valeurs numériques requises pour '{band_key}' et '{base_key}'.")
            st.stop()

        chosen = _choose_band(bands, x)
        if not chosen:
            st.error(f"Aucune bande trouvée pour {band_key}={x}.")
            st.stop()

        zone = str(values.get(factor_key))
        rates = chosen.get("rates", {})
        if zone not in rates:
            st.error(f"Taux indisponible pour {factor_key}={zone} dans la bande '{chosen.get('name','')}'.")
            st.stop()

        rate_per_unit = D(rates[zone])  # ex: kWhc / W
        raw = rate_per_unit * base_val
        detail = (
            f"bande '{chosen.get('name','')}' ({band_key}={x}) → taux[{zone}]={rate_per_unit} "
            f"× {base_key}={base_val}"
        )
        return {
            "name": out.get("name", "kwh_cumac"),
            "value": arrondir(raw, rnd),
            "raw": raw,
            "detail": detail,
            "unit": None,
        }

    # --- (4) zone_rate_times_band_factor : taux_zone × (facteur_bande optionnel) × (R optionnel) × (base_input optionnel) ---
    if kind == "zone_rate_times_band_factor":
        factor_key = block.get("factor_input")   # ex: zone_climatique
        zone_rates = block.get("zone_rates", {})
        zone = str(values.get(factor_key))
        if zone not in zone_rates:
            st.error(f"Taux de base indisponible pour zone={zone}.")
            st.stop()
        base_rate = D(zone_rates[zone])

        # facteur-bande (optionnel)
        band_factor = D(1)
        band_key = block.get("band_input")
        if band_key:
            bands = block.get("bands", [])
            x = _guess_number(values, band_key)
            if x is None:
                st.error(f"Valeur numérique requise pour '{band_key}'.")
                st.stop()
            chosen = _choose_band(bands, x)
            if not chosen:
                st.error(f"Aucune bande trouvée pour {band_key}={x}.")
                st.stop()
            band_factor = D(chosen.get("factor"))
            band_desc = f"× facteur_bande '{chosen.get('name','')}' (x={x} → {band_factor})"
        else:
            band_desc = ""

        # coefficient R (optionnel) : 1 entrée OU 2 entrées
        r_factor = D(1)
        r_inputs = block.get("r_inputs", [])
        r_table = block.get("r_table")
        r_desc = ""
        if r_table and r_inputs:
            if len(r_inputs) == 1:
                k1 = str(values.get(r_inputs[0]))
                if k1 not in r_table:
                    st.error(f"Coefficient R indisponible pour {r_inputs[0]}={k1}.")
                    st.stop()
                r_factor = D(r_table[k1])
                r_desc = f"× R[{r_inputs[0]}={k1}]={r_factor}"
            elif len(r_inputs) == 2:
                k1 = str(values.get(r_inputs[0]))
                k2 = str(values.get(r_inputs[1]))  # <-- fix ici: r_inputs (et non rinputs)
                if k1 not in r_table or k2 not in r_table[k1]:
                    st.error(f"Coefficient R indisponible pour {r_inputs[0]}={k1}, {r_inputs[1]}={k2}.")
                    st.stop()
                r_factor = D(r_table[k1][k2])
                r_desc = f"× R[{r_inputs[0]}={k1},{r_inputs[1]}={k2}]={r_factor}"
            else:
                st.error("r_inputs doit contenir 1 ou 2 clés.")
                st.stop()

        # multiplicateur de base (optionnel) : ex. nb_logements
        base_mult = D(1)
        base_input = block.get("base_input")
        if base_input:
            base_mult_val = _guess_number(values, base_input)
            if base_mult_val is None:
                st.error(f"Valeur numérique requise pour '{base_input}'.")
                st.stop()
            base_mult = base_mult_val

        raw = base_rate * band_factor * r_factor * base_mult
        detail = f"taux_zone[{zone}]={base_rate} {band_desc} {r_desc}"
        if base_input:
            detail += f" × {base_input}={base_mult}"
        return {
            "name": name,
            "value": arrondir(raw, rnd),
            "raw": raw,
            "detail": f"{detail}",
            "unit": None,
        }

    # --- (5) eta_zone_surface_composite : montant_base(ηs, usage, logement) × facteur_surface(S) × facteur_zone ---
    if kind == "eta_zone_surface_composite":
        eta_key = block.get("eta_input")
        usage_key = block.get("usage_input")
        log_key = block.get("logement_input")
        zone_key = block.get("zone_input")
        surf_key = block.get("surface_input")

        eta = _guess_number(values, eta_key)
        usage = str(values.get(usage_key))
        logement = str(values.get(log_key))
        zone = str(values.get(zone_key))
        S = _guess_number(values, surf_key)

        if None in (eta, usage, logement, zone, S):
            st.error("Paramètres manquants pour le calcul (ηs, usage, logement, zone, surface).")
            st.stop()

        # Bande ηs -> montant de base
        eta_bands = block.get("eta_bands", [])
        chosen_eta = _choose_band(eta_bands, eta)
        if not chosen_eta:
            st.error(f"Aucune bande ηs trouvée pour {eta_key}={eta}.")
            st.stop()
        amounts = chosen_eta.get("amounts", {})
        try:
            base_amount = D(amounts[logement][usage])
        except Exception:
            st.error("Montant de base introuvable pour la combinaison (ηs/usage/logement).")
            st.stop()

        # Facteur zone
        zf_raw = block.get("zone_factors", {}).get(zone, None)
        if zf_raw is None:
            st.error(f"Facteur de zone indisponible pour zone={zone}.")
            st.stop()
        zf = D(zf_raw)

        # Facteur surface (selon type de logement)
        surf_bands_all = block.get("surface_bands", {})
        bands = surf_bands_all.get(logement, [])
        chosen_s = _choose_band(bands, S)
        if not chosen_s:
            st.error(f"Aucune bande de surface trouvée pour S={S} (logement={logement}).")
            st.stop()
        sf = D(chosen_s.get("factor"))

        raw = base_amount * sf * zf
        detail = (
            f"bande_ηs '{chosen_eta.get('name','')}' → base={base_amount} "
            f"× facteur_surface '{chosen_s.get('name','')}'={sf} "
            f"× facteur_zone[{zone}]={zf}"
        )
        return {
            "name": out.get("name", "kwh_cumac"),
            "value": arrondir(raw, rnd),
            "raw": raw,
            "detail": detail,
            "unit": None,
        }

    # --- (6) delta_rate_by_factor_times_band : (rate[f_total] - rate[f_prev]) × facteur_bande(x) ---
    if kind == "delta_rate_by_factor_times_band":
        f_key = block.get("factor_input")         # ex: nb_sauts_total
        pf_key = block.get("prev_factor_input")   # ex: nb_sauts_etape1
        rates = block.get("rates", {})
        band_key = block.get("band_input")
        bands = block.get("bands", [])

        if not (f_key and pf_key and rates and band_key and bands):
            st.error("Configuration delta_rate_by_factor_times_band incomplète.")
            st.stop()

        f = str(values.get(f_key))
        pf = str(values.get(pf_key))
        if f not in rates or pf not in rates:
            st.error(f"Montants indisponibles pour factors total={f} / prev={pf}.")
            st.stop()

        x = _guess_number(values, band_key)
        if x is None:
            st.error(f"Valeur numérique requise pour '{band_key}'.")
            st.stop()
        chosen = _choose_band(bands, x)
        if not chosen:
            st.error(f"Aucune bande trouvée pour {band_key}={x}.")
            st.stop()

        base_total = D(rates[f])
        base_prev  = D(rates[pf])
        delta = base_total - base_prev
        if delta < 0:
            st.warning("Attention : le nombre de sauts cumulés est inférieur au nombre de sauts de l’étape 1.")
        factor = D(chosen.get("factor"))

        raw = delta * factor
        detail = (f"(base_total[{f}]={base_total} − base_etape1[{pf}]={base_prev}) "
                  f"× facteur_bande '{chosen.get('name','')}' (x={x} → {factor})")
        return {
            "name": out.get("name", "kwh_cumac"),
            "value": arrondir(raw, rnd),
            "raw": raw,
            "detail": detail,
            "unit": None,
        }

    # --- (7) scale : renvoie un multiplicateur (pour bonification de type 'applies: scale') ---
    if kind == "scale":
        # Deux manières : soit 'scale' constant, soit 'scale_table' 1D/2D
        scale = block.get("scale")
        scale_table = block.get("scale_table")
        sdesc = ""

        if scale is not None:
            s = D(scale)
            sdesc = f"multiplicateur={s}"
        elif scale_table:
            keys = []
            for fld in block.get("inputs", []):
                keys.append(str(values.get(fld["name"])))
            try:
                if len(keys) == 1:
                    s = D(scale_table[keys[0]])
                    sdesc = f"multiplicateur={s} (clé={keys[0]})"
                elif len(keys) == 2:
                    s = D(scale_table[keys[0]][keys[1]])
                    sdesc = f"multiplicateur={s} (clés={keys[0]},{keys[1]})"
                else:
                    st.error("La table de multiplicateur attend 1 ou 2 clés.")
                    st.stop()
            except Exception:
                st.error("Combinaison introuvable dans la table de multiplicateur.")
                st.stop()
        else:
            s = D(1)

        return {
            "name": out.get("name", "multiplicateur"),
            "value": D(0),
            "raw": D(0),
            "detail": sdesc,
            "unit": None,
            "scale": s,
        }

    st.error(f"Type de calcul non pris en charge: {kind}")
    st.stop()

# ------------------------------
# UI principale
# ------------------------------
def main():
    st.set_page_config(page_title="Calculette CUMAC — Pro", layout="centered")
    st.title("Calculette CUMAC — Pro")

    fiches = load_fiches()
    if not fiches:
        st.warning("Aucune fiche trouvée dans le dossier 'params/'.")
        st.stop()

    titles = [f"{f.get('code','???')} — {f.get('title','(sans titre)')}" for f in fiches]
    sel = st.selectbox("Fiche d’opération", titles, index=0)
    fiche = fiches[titles.index(sel)]

    st.markdown(f"**Description :** {fiche.get('description','')}")
    if fiche.get("eligibility_details"):
        with st.expander("Détails / Éligibilité"):
            st.markdown(fiche["eligibility_details"])

    st.subheader("Paramètres")
    values = {}
    for field in fiche.get("inputs", []):
        k, v = render_input(field)
        values[k] = v

    # Résolution de variante éventuelle
    calc_spec = fiche.get("calcul_standard", {})
    calc_block = _pick_variant(calc_spec, values)

    st.subheader("Résultat")
    std = compute_block(calc_block, values)
    st.markdown(f"- **Standard (kWh cumac)** : {std['value']:,}".replace(",", " "))
    st.caption(f"Détail : {std['detail']}")

    total = std["value"]
    total_breakdown = [("Standard", std["value"])]

    # Bonifications (optionnelles) : additive / replacement / scale
    bonuses = fiche.get("bonifications", []) or []
    active_bonuses = []
    if bonuses:
        with st.expander("Bonifications (optionnel)"):
            for idx, b in enumerate(bonuses):
                use = st.checkbox(b.get("name", f"Bonus {idx+1}"),
                                  value=b.get("enabled_by_default", False),
                                  key=f"bonus_{idx}")
                if use:
                    for field in b.get("inputs", []):
                        k, v = render_input(field)
                        values[k] = v
                    active_bonuses.append(b)

    for b in active_bonuses:
        applies = b.get("applies", "additive")
        if applies == "scale":
            res = compute_block(b, values)  # renverra 'scale'
            mult = res.get("scale", D(1))
            total = arrondir(D(total) * mult, 0)
            total_breakdown.append((b.get("name","Bonus (×)"), f"× {mult}"))
            st.markdown(f"- **{b.get('name','Bonus')}** : × {mult}")
            st.caption(f"Détail : {res.get('detail','')}")
        else:
            res = compute_block(b, values)
            if applies == "replacement":
                total = res["value"]
                total_breakdown = [(b.get('name','Bonus'), res["value"])]
            else:
                total += res["value"]
                total_breakdown.append((b.get('name','Bonus'), res["value"]))
            st.markdown(f"- **{b.get('name','Bonus')} (kWh cumac)** : {res['value']:,}".replace(",", " "))
            st.caption(f"Détail : {res['detail']}")

    st.markdown(f"**Total kWh cumac** : {total:,}".replace(",", " "))

    st.divider()
    st.subheader("Conversion en €")
    # Conversion en €/MWhc
    ppk_mwh = st.number_input("Prix (€/MWh cumac)", min_value=0.0, max_value=100000.0, value=0.0, step=0.1)
    total_mwh = float(total) / 1000.0
    montant_eur = total_mwh * ppk_mwh
    st.markdown(f"**Montant estimé (€)** : {montant_eur:,.2f}".replace(",", " "))

    with st.expander("Détail de calcul (récap)"):
        for label, v in total_breakdown:
            st.write(f"- {label} : {v:,} kWhc".replace(",", " "))
        st.write(
            f"Total : {total:,} kWhc = {total_mwh:,.3f} MWhc × {ppk_mwh} €/MWhc = {montant_eur:,.2f} €"
            .replace(",", " ")
        )

    meta = fiche.get("meta", {})
    if meta:
        with st.expander("Métadonnées"):
            st.json(meta)

if __name__ == "__main__":
    main()
