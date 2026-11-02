"""Reconstruct C0 valence radial functions from the frozen ethane scan.

No SCF, network inference, optimization, or production-code changes are needed.
Run from the repository root with a Python containing NumPy/SciPy/Matplotlib.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import time

import numpy as np
from scipy.integrate import quad

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "nnao-ethane-torsion-20261018"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ANGLES = np.arange(0, 121, 5)
METHODS = {"MACE": "MACE-update3351", "Free": "free-contraction-best"}
COLORS = {"MACE": "#D87632", "Free": "#28699C"}
BOHR_ANGSTROM = 0.529177210903
R = np.linspace(0, 8, 1601)  # Bohr; integration uses analytic infinite-range moments.


def moment_matrix(l, alpha, order=0):
    power = l + 1.5 + order / 2
    return math.gamma(power) / (2 * (alpha[:, None] + alpha[None, :]) ** power)


def normalized_weights(shell):
    l = int(shell[0])
    rows = np.asarray(shell[1:], dtype=np.float64)
    assert rows.shape == (6, 2) and np.all(rows[:, 0] > 0)
    alpha, raw = rows.T
    # Same radial convention as integrals/basis/normalization.py.
    n = l + 1.5
    weights = raw * np.sqrt(2 * (2 * alpha) ** n / math.gamma(n))
    weights /= np.sqrt(weights @ moment_matrix(l, alpha) @ weights)
    return alpha, raw, weights


def evaluate(l, alpha, weights, radius):
    r = np.asarray(radius)
    return r ** l * (np.exp(-r[..., None] ** 2 * alpha) @ weights)


def radial_norm(l, alpha, weights):
    return float(np.sqrt(max(0.0, weights @ moment_matrix(l, alpha) @ weights)))


def dump_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.perf_counter()
    scan = json.loads((SOURCE / "scan-summary.json").read_text())
    assert scan["status"] == "complete" and len(scan["points"]) == 25
    data = {method: {l: [] for l in (0, 1)} for method in METHODS}
    provenance, radial_rows, coefficient_rows, metrics = [], [], [], []
    normalization_error, analytic_quadrature_error = [], []
    reference, fixed_alpha = {}, {}
    checkpoint = None
    for angle in ANGLES:
        folder = SOURCE / f"angle-{angle:03d}"
        path = folder / "frozen-bases.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        config = json.loads((folder / "configuration.json").read_text())
        geometry = json.loads((folder / "structure.json").read_text())
        assert config["fixed_exponents"] and config["fixed_core"]
        assert config["angle_deg"] == angle
        assert config["checkpoint"]["update_step"] == 3351
        if checkpoint is None:
            checkpoint = config["checkpoint"]
            original_geometry = geometry
        assert config["checkpoint"]["parameter_sha256"] == checkpoint["parameter_sha256"]
        provenance.append({"path": str(path.relative_to(OUT.parent.parent)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for method, label in METHODS.items():
            record = next(record for record in records if record["basis"] == label)
            assert record["nao"] == 44 and record["atom_labels"][0] == "C0"
            np.testing.assert_allclose(record["coords_angstrom"], geometry["coords_angstrom"], atol=1e-14)
            assert [shell[0] for shell in record["atom_shells"][0]] == [0, 0, 1, 2]
            for l, index in [(0, 1), (1, 2)]:
                alpha, raw, weights = normalized_weights(record["atom_shells"][0][index])
                if l not in fixed_alpha:
                    fixed_alpha[l] = alpha
                    # Choose a positive main radial-probability lobe at MACE 0 degrees.
                    test = R * evaluate(l, alpha, weights, R)
                    weights *= 1 if test[np.argmax(abs(test))] > 0 else -1
                    reference[l] = weights.copy()
                np.testing.assert_array_equal(alpha, fixed_alpha[l])
                overlap = weights @ moment_matrix(l, alpha) @ reference[l]
                sign = 1 if overlap >= 0 else -1
                weights *= sign
                radial = evaluate(l, alpha, weights, R)
                data[method][l].append({"weights": weights, "radial": radial})
                norm2 = weights @ moment_matrix(l, alpha) @ weights
                mean = float(weights @ moment_matrix(l, alpha, 1) @ weights)
                rms = float(np.sqrt(weights @ moment_matrix(l, alpha, 2) @ weights))
                numerical_norm = quad(lambda r: r*r * float(evaluate(l, alpha, weights, r))**2,
                                      0, np.inf, epsabs=1e-11, epsrel=1e-11)[0]
                numerical_mean = quad(lambda r: r**3 * float(evaluate(l, alpha, weights, r))**2,
                                      0, np.inf, epsabs=1e-11, epsrel=1e-11)[0]
                normalization_error.append(abs(norm2 - 1))
                analytic_quadrature_error.extend([abs(norm2-numerical_norm), abs(mean-numerical_mean)])
                delta = weights - data[method][l][0]["weights"]
                metrics.append({"method": method, "angle_deg": int(angle), "channel": "sp"[l],
                    "mean_radius_bohr": mean, "mean_radius_angstrom": mean*BOHR_ANGSTROM,
                    "rms_radius_bohr": rms, "change_from_0_L2": radial_norm(l, alpha, delta),
                    "overlap_with_MACE_0": abs(float(overlap)), "phase_sign": sign,
                    "energy_hartree": record["native_energy_hartree"]})
                for p, (a, c, w) in enumerate(zip(alpha, raw, weights)):
                    coefficient_rows.append({"method": method, "angle_deg": int(angle),
                        "channel": "sp"[l], "primitive": p, "exponent_bohr_minus2": a,
                        "raw_coefficient": c, "aligned_normalized_radial_weight": w})
                for r, value in zip(R, radial):
                    radial_rows.append({"method": method, "angle_deg": int(angle),
                        "channel": "sp"[l], "radius_bohr": r, "R_bohr_minus_1p5": value,
                        "rR_bohr_minus_0p5": r*value, "radial_probability_bohr_minus1": (r*value)**2})
    assert max(normalization_error) < 1e-12
    assert max(analytic_quadrature_error) < 1e-10
    for method in METHODS:
        for l in (0, 1):
            assert radial_norm(l, fixed_alpha[l], data[method][l][-1]["weights"] -
                               data[method][l][0]["weights"]) < 5e-6

    differences = []
    for i, angle in enumerate(ANGLES):
        for l in (0, 1):
            differences.append({"angle_deg": int(angle), "channel": "sp"[l],
                "MACE_minus_Free_L2": radial_norm(l, fixed_alpha[l],
                    data["MACE"][l][i]["weights"] - data["Free"][l][i]["weights"])})
    dump_csv(OUT / "radial-functions.csv", radial_rows)
    dump_csv(OUT / "radial-metrics.csv", metrics)
    dump_csv(OUT / "radial-coefficients.csv", coefficient_rows)
    dump_csv(OUT / "method-differences.csv", differences)
    np.savez_compressed(OUT / "radial-functions.npz", radius_bohr=R, angles_deg=ANGLES,
        **{f"{method.lower()}_{'sp'[l]}": np.array([r["radial"] for r in data[method][l]])
           for method in METHODS for l in (0, 1)})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "axes.labelcolor": "#26313A", "text.color": "#26313A"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for l in (0, 1):
        for method in METHODS:
            for angle, style in [(0, "-"), (30, "--"), (60, ":")]:
                i = int(angle/5)
                radial = data[method][l][i]["radial"]
                axes[0, l].plot(R, radial, color=COLORS[method], ls=style, lw=1.8)
                if angle:
                    axes[1, l].plot(R, radial-data[method][l][0]["radial"],
                                    color=COLORS[method], ls=style, lw=2)
        axes[0, l].set_title(f"C0 valence {'s' if l == 0 else 'p'} radial function")
        axes[0, l].set_ylabel(r"$R_l(r)$ ($a_0^{-3/2}$)")
        axes[1, l].set_ylabel(r"$R_l(r;\phi)-R_l(r;0)$ ($a_0^{-3/2}$)")
        axes[1, l].set_xlabel(r"Distance from C0, $r$ ($a_0$)")
        for ax in axes[:, l]:
            ax.set_xlim(0, 6)
            ax.axhline(0, color="#AEB5BA", lw=.7)
            ax.grid(color="#E6E9EB", lw=.6)
        axes[1, l].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    handles = [Line2D([], [], color=COLORS[m], lw=2, label=m) for m in METHODS]
    handles += [Line2D([], [], color="#49535C", ls=s, lw=1.6, label=f"{a} degrees")
                for a, s in [(0, "-"), (30, "--"), (60, ":")]]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(.5, .94))
    fig.suptitle("Ethane torsion: carbon valence radial functions", fontsize=15, y=.995)
    fig.text(.09, .025, "C0; 44-AO NNAO; fixed 6s/6p primitive pools; MACE update 3351 vs free contraction.\n"
             "Normalized radial basis functions, not occupied molecular orbitals; phases aligned to MACE at 0 degrees.",
             fontsize=9, color="#59636B")
    fig.subplots_adjust(left=.09, right=.98, top=.86, bottom=.13, wspace=.28, hspace=.25)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"carbon-radial-functions.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), sharex=True)
    for l in (0, 1):
        for method in METHODS:
            rows = [x for x in metrics if x["method"] == method and x["channel"] == "sp"[l]]
            means = np.array([x["mean_radius_bohr"] for x in rows])
            axes[0, l].plot(ANGLES, means, color=COLORS[method], marker="o" if method == "Free" else "s",
                            markersize=3.2, lw=1.8, label=method)
            axes[1, l].plot(ANGLES, (means-means[0])*1000, color=COLORS[method],
                            marker="o" if method == "Free" else "s", markersize=3.2, lw=1.8)
        axes[0, l].set_title(f"C0 valence {'sp'[l]}: mean radial distance")
        axes[0, l].set_ylabel(r"$\langle r\rangle$ ($a_0$)")
        axes[0, l].ticklabel_format(axis="y", useOffset=False)
        axes[1, l].set_ylabel(r"$\langle r\rangle_\phi-\langle r\rangle_0$ ($10^{-3}a_0$)")
        axes[1, l].set_xlabel("Methyl rotation (degrees)")
        axes[1, l].axhline(0, color="#AEB5BA", lw=.7)
        for ax in axes[:, l]:
            ax.grid(color="#E6E9EB", lw=.6)
            ax.set_xticks(np.arange(0, 121, 30))
            ax.axvline(60, color="#AEB5BA", lw=.7, ls=":")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Radial response along the complete 25-point torsion scan", fontsize=14)
    fig.text(.1, .025, r"$\langle r\rangle=\int_0^\infty r^3|R_l(r)|^2\,dr$; evaluated analytically, with independent quadrature checks."
             "\nMean radius describes the basis function; it is not an atomic charge or electron-density radius.",
             fontsize=9, color="#59636B")
    fig.subplots_adjust(left=.1, right=.98, top=.89, bottom=.14, wspace=.3, hspace=.24)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"carbon-radial-response.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)

    export_html(data, metrics, original_geometry)
    summary = {"status": "complete", "source": str(SOURCE), "atom": "C0", "atom_index": 0,
        "coords_angstrom": original_geometry["coords_angstrom"][0], "angles_deg": ANGLES.tolist(),
        "method": "DF-RHF", "auxbasis": "def2-universal-jkfit", "contracted_nao": 44,
        "fixed_exponents": True, "fixed_core": True, "checkpoint_update": 3351,
        "checkpoint_parameter_sha256": checkpoint["parameter_sha256"],
        "normalization_max_error": max(normalization_error),
        "analytic_vs_quadrature_max_error": max(analytic_quadrature_error),
        "elapsed_seconds": time.perf_counter()-started, "sources": provenance,
        "notes": ["No new SCF, training or finite-difference evaluation.",
                  "Raw valence-s basis function; not core-orthogonalized.",
                  "All p components share one radial function.",
                  "Free contraction endpoints are finite-budget minima, not proved global minima."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["# 乙烷扭转过程中 C0 的价层径向函数", "",
        "比较原 44 AO NNAO 的 MACE update 3351 与逐构型自由收缩。固定指数和内层，"
        "DF-RHF/def2-universal-jkfit；25 个刚性构型，0–120°，间隔 5°。",
        "C0 是输入的第一个碳，坐标 (0, 0, +0.7629881884) Å；H5/H6/H7 绕 C–C 轴旋转。",
        "价层 s/p 各有 6 个 primitive，分别收缩为 1 个径向通道。固定 core s 不在图中。", "",
        "## 定义与约定", "",
        "`chi_lm = R_l(r) Y_lm`，球谐函数归一化；`R_l = r^l sum_p w_p exp(-alpha_p r^2)`，"
        "且 `integral r^2 |R_l|^2 dr = 1`。w 包含 primitive 和收缩归一化。"
        "采用与生产积分器相同的归一化公式，所有曲线对齐到 MACE 0° 的正主瓣。",
        "`<r> = integral r^3 |R_l|^2 dr`。半径为 Bohr (a0)，R 为 a0^(-3/2)。"
        "CSV 同时提供 rR 和径向概率 r^2|R|^2。p_x/p_y/p_z 共用该径向部分。",
        "这是基函数形状，不是占据分子轨道或电子密度；s 保持原始价层基函数，没有额外与 core 正交化。", "",
        "## 结果", "", "| 通道 | 方法 | <r>(0°) / a0 | <r>(60°) / a0 | 差值 / a0 |",
        "|---|---|---:|---:|---:|"]
    for l in (0, 1):
        for method in METHODS:
            rows = [x for x in metrics if x["method"] == method and x["channel"] == "sp"[l]]
            v0, v60 = rows[0]["mean_radius_bohr"], rows[12]["mean_radius_bohr"]
            lines.append(f"| {'sp'[l]} | {method} | {v0:.10f} | {v60:.10f} | {v60-v0:+.10f} |")
    lines += ["", "本构型扫描中 MACE 向 60° 预测径向收缩，自由变分端点的平均半径略增。"
              "该差异说明本例的基函数环境响应不同，不能仅凭径向曲线把能量差归因到一个 C 的 s 或 p 通道。", "",
              "## 文件与复现", "", "- `carbon-radial-functions.png/.pdf`：0/30/60° 的 R 和相对 0° 的变化。",
              "- `carbon-radial-response.png/.pdf`：全部 25 个角度的平均半径。",
              "- `carbon-radial-interactive.html`：拖动角度，比较 R、rR 或径向概率，并显示对应分子。",
              "- `radial-functions.csv/.npz`、`radial-coefficients.csv`、`radial-metrics.csv`：函数、系数和指标。",
              "- `summary.json`：源文件 SHA256、检查点、校验误差和本地绘图耗时。", "",
              "命令：`/opt/anaconda3/bin/python artifacts/nnao-ethane-radial-20261019/plot_radial.py`。",
              "本地 CPU float64 后处理；没有重新运行 SCF、MACE、训练或有限差分。",
              f"归一化最大误差 {max(normalization_error):.3g}；解析矩与独立数值积分最大差 {max(analytic_quadrature_error):.3g}。",
              "原始自由收缩是有限预算优化；多起点一致不构成全局最优证明。"]
    (OUT / "README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, indent=2))


def export_html(data, metrics, geometry):
    # Self-contained SVG viewer; no external assets, libraries or network access.
    payload = {"angles": ANGLES.tolist(), "r": R[::4].round(8).tolist(),
        "curves": {f"{m}_{l}": np.array([x["radial"][::4] for x in data[m][l]]).tolist()
                   for m in METHODS for l in (0, 1)}, "metrics": metrics,
        "geometry": geometry, "colors": COLORS}
    template = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>乙烷 C0 径向函数</title>
<style>body{font:16px system-ui,sans-serif;color:#26313a;background:#fafbfc;margin:0}main{max-width:1150px;margin:auto;padding:24px}h1{font-size:24px}p{line-height:1.6}.controls{display:flex;align-items:center;gap:20px;flex-wrap:wrap;padding:16px 0}input{width:300px;max-width:65vw}select,button{font:inherit;padding:6px}.plots{display:grid;grid-template-columns:1fr 1fr;gap:20px}.plot{background:white;border:1px solid #e0e5e9;border-radius:8px;padding:10px}.plot h2{font-size:17px;margin:8px}svg{width:100%;height:auto}svg text{font:13px system-ui,sans-serif;fill:#49535c}.sub{color:#59636b;font-size:14px}.legend span{margin-right:20px}table{border-collapse:collapse;width:100%;max-width:700px}td,th{text-align:left;padding:8px;border-bottom:1px solid #dce1e5}.top{display:flex;align-items:center;gap:24px;flex-wrap:wrap}#molecule{width:220px;height:180px}.value{font-variant-numeric:tabular-nums}@media(max-width:700px){main{padding:14px}.plots{grid-template-columns:1fr}h1{font-size:21px}.controls{gap:10px}}</style>
<main><h1>乙烷扭转：C0 的价层 s、p 径向函数</h1>
<div class="top"><div><p>同一 44 AO 基组、固定指数和内层。对比 MACE update 3351 与逐构型自由变分。<br>0° 为初始交错构型，60° 为本扫描的重叠构型；旋转的是 H5/H6/H7。</p><p class="sub">绘制原始归一化价层基函数；不含固定 core s。三个 p 分量共享径向部分。<br>全部角度直接读取已有导出基组；未重跑 SCF 或训练。</p></div><svg id="molecule" viewBox="0 0 220 180" aria-label="沿 C-C 轴观察的乙烷示意图"></svg></div>
<div class="controls"><label>扭转角 <input id="angle" type="range" min="0" max="24" value="0" step="1"> <b id="angleValue">0°</b></label><button id="play">播放</button><label>显示 <select id="quantity"><option value="R">径向函数 R(r)</option><option value="u">约化径向函数 rR(r)</option><option value="P">径向概率 r²|R(r)|²</option></select></label></div>
<p class="legend"><span style="color:#28699c">━ 自由变分</span><span style="color:#d87632">┄ MACE</span><span class="sub">半径 r：Bohr (a₀)；图中范围 0–6 a₀。</span></p>
<div class="plots"><div class="plot"><h2>价层 s</h2><svg id="s" viewBox="0 0 530 305"></svg></div><div class="plot"><h2>价层 p</h2><svg id="p" viewBox="0 0 530 305"></svg></div><div class="plot"><h2>s：相对各自 0° 的变化</h2><svg id="ds" viewBox="0 0 530 305"></svg></div><div class="plot"><h2>p：相对各自 0° 的变化</h2><svg id="dp" viewBox="0 0 530 305"></svg></div></div>
<p id="unit" class="sub"></p><table><thead><tr><th>通道</th><th>方法</th><th>平均半径 / a₀</th><th>相对 0° 变化 / a₀</th></tr></thead><tbody id="values" class="value"></tbody></table>
<p class="sub">平均半径由 ∫r³|R|²dr 定义，解析积分至无穷远。图为基函数而非电子密度或占据分子轨道。s 未额外与 core 正交化；全部符号统一到 MACE 0° 的主瓣。曲线只反映该乙烷扫描；自由变分端点不构成严格全局极限。</p></main>
<script>const D=__DATA__;const ns="http://www.w3.org/2000/svg";let timer;
function el(tag,attrs,text){let e=document.createElementNS(ns,tag);for(let k in attrs)e.setAttribute(k,attrs[k]);if(text!==undefined)e.textContent=text;return e}
function transform(v,k,q){return q==='u'?v*D.r[k]:q==='P'?(v*D.r[k])**2:v}
function values(m,l,i,q,delta){let v=D.curves[m+'_'+l][i],b=D.curves[m+'_'+l][0];return v.map((y,k)=>transform(y,k,q)-(delta?transform(b[k],k,q):0))}
function chart(id,l,i,q,delta){let svg=document.getElementById(id);svg.replaceChildren();let all=[];for(let m of ['MACE','Free'])for(let j=0;j<25;j++)all.push(...values(m,l,j,q,delta).filter((v,k)=>D.r[k]<=6));let lo=Math.min(0,...all),hi=Math.max(0,...all),pad=(hi-lo)*.08||1;lo-=pad;hi+=pad;let X=r=>65+r/6*440,Y=v=>250-(v-lo)/(hi-lo)*215;for(let k=0;k<=6;k++){svg.append(el('line',{x1:X(k),x2:X(k),y1:35,y2:250,stroke:'#e6e9eb'}));svg.append(el('text',{x:X(k),y:273,'text-anchor':'middle'},k))}for(let k=0;k<=4;k++){let v=lo+(hi-lo)*k/4;svg.append(el('line',{x1:65,x2:505,y1:Y(v),y2:Y(v),stroke:'#e6e9eb'}));svg.append(el('text',{x:58,y:Y(v)+4,'text-anchor':'end'},Math.abs(v)<.01?v.toExponential(1):v.toFixed(2)))}svg.append(el('line',{x1:65,x2:505,y1:Y(0),y2:Y(0),stroke:'#aeb5ba'}));for(let m of ['Free','MACE']){let v=values(m,l,i,q,delta),pts=[];for(let k=0;k<D.r.length&&D.r[k]<=6;k++)pts.push(X(D.r[k])+','+Y(v[k]));svg.append(el('polyline',{points:pts.join(' '),fill:'none',stroke:D.colors[m],'stroke-width':2.4,'stroke-dasharray':m==='MACE'?'7 4':''}))}svg.append(el('text',{x:285,y:299,'text-anchor':'middle'},'r / a₀'))}
function molecule(i){let svg=document.getElementById('molecule');svg.replaceChildren();let g=D.geometry.coords_angstrom.map(v=>v.slice()),a=D.angles[i]*Math.PI/180;for(let k of D.geometry.rotating_hydrogen_indices){let [x,y,z]=g[k];g[k]=[x*Math.cos(a)-y*Math.sin(a),x*Math.sin(a)+y*Math.cos(a),z]}for(let k=2;k<8;k++){let p=g[k],front=p[2]>0;svg.append(el('line',{x1:110,y1:90,x2:110+p[0]*62,y2:90-p[1]*62,stroke:front?'#28699c':'#9ba5ac','stroke-width':front?3:2,'stroke-dasharray':front?'':'5 3'}));svg.append(el('circle',{cx:110+p[0]*62,cy:90-p[1]*62,r:9,fill:front?'#dbe9f4':'#f1f2f3',stroke:'#68747e'}))}svg.append(el('circle',{cx:110,cy:90,r:12,fill:'#28699c'}));svg.append(el('text',{x:110,y:115,'text-anchor':'middle'},'C0（前） / C1（后）'))}
function update(){let i=+document.getElementById('angle').value,q=document.getElementById('quantity').value;document.getElementById('angleValue').textContent=D.angles[i]+'°';for(let [id,l,d] of [['s',0,false],['p',1,false],['ds',0,true],['dp',1,true]])chart(id,l,i,q,d);document.getElementById('unit').textContent=q==='R'?'纵轴 R 的单位：a₀⁻³ᐟ²。':q==='u'?'纵轴 rR 的单位：a₀⁻¹ᐟ²。':'纵轴径向概率的单位：a₀⁻¹；其径向积分等于 1。';document.getElementById('values').innerHTML=[0,1].flatMap(l=>['Free','MACE'].map(m=>{let row=D.metrics.find(x=>x.method===m&&x.channel==='sp'[l]&&x.angle_deg===D.angles[i]),base=D.metrics.find(x=>x.method===m&&x.channel==='sp'[l]&&x.angle_deg===0);return '<tr><td>'+('sp'[l])+'</td><td>'+m+'</td><td>'+row.mean_radius_bohr.toFixed(8)+'</td><td>'+(row.mean_radius_bohr-base.mean_radius_bohr).toFixed(8)+'</td></tr>'})).join('');molecule(i)}document.getElementById('angle').oninput=update;document.getElementById('quantity').onchange=update;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放'}else{timer=setInterval(()=>{let s=document.getElementById('angle');s.value=(+s.value+1)%25;update()},350);document.getElementById('play').textContent='暂停'}};update();</script></html>'''
    (OUT / "carbon-radial-interactive.html").write_text(template.replace("__DATA__", json.dumps(payload)))


if __name__ == "__main__":
    main()
