"""Summarize complete experiments without selecting seeds by test scores."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from .common import read_json, write_json
from .text import spans


def stats(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values)}


def report(run_dir):
    root = Path(run_dir).resolve()
    plan, status = read_json(root / "plan.json"), read_json(root / "status.json")
    cfg, identity = plan["config"], plan["plan_id"]
    if not status.get("complete"):
        raise ValueError("Run is incomplete; refusing to present a partial run as a final comparison")
    seeds, arms = cfg["ner"]["seeds"], plan["arms"]
    output = root / "report"
    output.mkdir(exist_ok=True)
    rows, summary, by_arm = [], {}, {}
    for arm in arms:
        by_arm[arm] = {}
        for seed in seeds:
            folder = root / "ner" / arm / str(seed)
            value = read_json(folder / "test.json")
            if value["fingerprint"] != identity:
                raise ValueError("Mixed experiment fingerprints in report")
            completion = read_json(folder / "complete.json")
            metrics = value["metrics"]
            by_arm[arm][seed] = metrics["ALL"]["f1"]
            rows.append({"arm": arm, "seed": seed, **metrics["ALL"],
                         "dev_f1": completion["best_f1"], "epochs": completion["epoch"],
                         "seconds": completion["seconds"], "peak_vram_bytes": completion["peak_vram_bytes"],
                         "parameters": completion["parameters"],
                         **{kind + "_f1": metrics[kind]["f1"] for kind in ("PER", "ORG", "LOC", "MISC")}})
        summary[arm] = stats(list(by_arm[arm].values()))
        if arm != "static":
            summary[arm]["paired_delta_vs_static"] = stats([by_arm[arm][s] - by_arm["static"][s] for s in seeds])
        # Fixed first seed avoids choosing the best test run for qualitative analysis.
        errors = []
        for row in read_json(root / "ner" / arm / str(seeds[0]) / "predictions.json"):
            gold, pred = spans(row["tags"]), spans(row["predicted"])
            if gold == pred:
                continue
            category = "boundary_or_detection"
            if {(a, b) for a, b, _ in gold} == {(a, b) for a, b, _ in pred}:
                category = "entity_type"
            errors.append({"id": row["id"], "doc_id": row["doc_id"], "tokens": row["tokens"],
                           "gold": sorted(gold), "predicted": sorted(pred), "automatic_category": category,
                           "linguistic_analysis": ""})
            if len(errors) == 20:
                break
        write_json(output / f"errors_{arm}.json", {"seed": seeds[0], "selection": "first 20 erroneous test sentences in original order", "errors": errors})
    summary["depth2_minus_depth1"] = stats([by_arm["depth2"][s] - by_arm["depth1"][s] for s in seeds])
    write_json(output / "summary.json", summary)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    resources = {f"depth{d}": read_json(root / f"lm_depth{d}" / "complete.json") for d in (1, 2)}
    write_json(output / "lm_resources.json", resources)
    label = "SYNTHETIC SMOKE TEST — NOT BENCHMARK RESULTS" if cfg.get("smoke") else "CoNLL-2003 NER Results"
    lines = [f"# {label}", "", "| Arm | Test entity F1, mean ± sample std | Seeds | Paired Δ vs static |",
             "|---|---:|---:|---:|"]
    for arm in arms:
        s = summary[arm]
        delta = s.get("paired_delta_vs_static", {}).get("mean", 0)
        lines.append(f"| {arm} | {s['mean']:.2f} ± {s['std']:.2f} | {s['n']} | {delta:+.2f} |")
    lines += ["", "## 문헌 참고값 (본 실험의 측정값 아님)", "",
              "아래 점수는 논문에 보고된 값이다. 학습 데이터량·모델 규모·학습 조건이 달라 직접 재현 결과로 해석하지 않는다.", "",
              "| 참고 연구 | CoNLL-2003 test F1 |",
              "|---|---:|",
              "| [ELMo baseline](https://aclanthology.org/N18-1202.pdf) | 90.15 |",
              "| [ELMo + baseline](https://aclanthology.org/N18-1202.pdf) | 92.22 ± 0.10 |",
              "| [Lample et al. 2016](https://aclanthology.org/N16-1030.pdf) | 90.94 |", "",
              "## 비교 조건과 한계", "",
              "- 직접 학습은 1B Word의 축소 코퍼스·50k 출력 어휘·축소 LSTMP를 사용한다.",
              ("- 선택적으로 실행한 공개 ELMo 비교군은 NER LM 적응 학습을 생략한다." if "public" in arms else
               "- 공개 ELMo 모델은 다운로드하거나 실행하지 않는다. 원문 수치는 문헌 비교에만 사용한다."),
              "- 모든 임베딩은 문장마다 상태를 초기화한다. 원래 참조 구현의 배치 간 state carry와 다르다.",
              "- 긴 LM 문장을 고정 길이 조각으로 나누고 경계를 추가한다. NER 문장은 자르지 않는다.",
              "- SENNA는 원형 일치 후 소문자·숫자 정규화로 조회한다. coverage는 manifest에 기록한다.",
              "- downstream 표준편차는 고정된 사전학습 checkpoint 아래의 변동성이다.",
              "- depth 변화는 파라미터 수도 바꾸므로 순수한 깊이만의 인과 효과로 해석하지 않는다.",
              "- static 대비 차이에는 추가 모델 용량·사전학습 효과도 포함된다.",
              "- test는 전체 dev 기반 학습 완료 후 평가했다. 모델별 토큰 accuracy 대신 entity F1을 사용했다.",
              "- errors_*.json의 linguistic_analysis는 자동으로 채우지 않았다. 실제 예측을 읽고 분석해야 한다.", "",
              f"총 실행 시간: {(status['seconds'] + plan['pilot_seconds']) / 3600:.3f}시간. 데이터 준비 시간 제외.",
              f"실제 GPU: {plan['environment']['gpu']}. 실행 ID: `{identity}`.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    plot_curves(root, output, arms, seeds)
    print(f"Report: {output / 'report.md'}")
    return summary


def plot_curves(root, output, arms, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for depth in (1, 2):
        records = [json.loads(line) for line in (root / f"lm_depth{depth}" / "metrics.jsonl").read_text().splitlines()]
        records = [r for r in records if "train_nll" in r]
        axes[0].plot([r["tokens"] for r in records], [r["train_nll"] for r in records], label=f"depth {depth}")
    axes[0].set(xlabel="Training corpus words (directions counted once)", ylabel="Train NLL")
    for arm in arms:
        records = [json.loads(line) for line in (root / "ner" / arm / str(seeds[0]) / "metrics.jsonl").read_text().splitlines()]
        axes[1].plot([r["epoch"] for r in records], [r["dev"]["ALL"]["f1"] for r in records], label=arm)
    axes[1].set(xlabel=f"Epoch (fixed seed {seeds[0]})", ylabel="Dev entity F1")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    report(args.run_dir)


if __name__ == "__main__":
    main()
