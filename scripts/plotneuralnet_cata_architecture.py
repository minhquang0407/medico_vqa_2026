"""Generate the CATA architecture figure as a fixed-coordinate 2D TikZ diagram.

The layout follows a clean two-stream system diagram:
- Vision + topology stream on the left.
- Language generation stream on the right.
- Fixed coordinates for stable, aligned rendering.

Usage from this repository root:

    python scripts/plotneuralnet_cata_architecture.py \
        --output paper_assets/figures/cata_architecture

Then compile the generated TeX file yourself:

    pdflatex paper_assets/figures/cata_architecture.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path


CATA_ARCH_TEX = r"""\documentclass[tikz,border=7pt]{standalone}
\usepackage{amsmath}
\usetikzlibrary{arrows.meta,calc,backgrounds,shapes.geometric}

\definecolor{cataTeal}{HTML}{67DAD5}
\definecolor{cataLine}{HTML}{4B5563}
\definecolor{cataVit}{HTML}{E8EEFF}
\definecolor{cataVitBorder}{HTML}{6675FF}
\definecolor{cataVisual}{HTML}{ECF4FF}
\definecolor{cataVisualBorder}{HTML}{4B91F1}
\definecolor{cataTda}{HTML}{FFF1F8}
\definecolor{cataTdaBorder}{HTML}{FF6AA9}
\definecolor{cataGreen}{HTML}{D7F7E8}
\definecolor{cataGreenBorder}{HTML}{20C985}
\definecolor{cataQwen}{HTML}{F1E7FF}
\definecolor{cataQwenBorder}{HTML}{9867FF}
\definecolor{cataAdapter}{HTML}{FFF4D8}
\definecolor{cataAdapterBorder}{HTML}{F2A51A}
\definecolor{cataEvidence}{HTML}{FFE8E8}
\definecolor{cataEvidenceBorder}{HTML}{FF6666}
\definecolor{cataTask}{HTML}{E8F7FF}
\definecolor{cataTaskBorder}{HTML}{37B6EF}

\pgfdeclarelayer{background}
\pgfsetlayers{background,main}

\begin{document}
\begin{tikzpicture}[
    x=1cm,
    y=1cm,
    font=\sffamily,
    >=Latex,
    flow/.style={-Latex, line width=0.95pt, draw=cataLine},
    box/.style={
        rounded corners=6pt,
        line width=1.0pt,
        align=center,
        font=\sffamily\small,
        minimum height=0.95cm,
        inner sep=4pt
    },
    vitbox/.style={box, fill=cataVit, draw=cataVitBorder, minimum width=1.65cm},
    greenpara/.style={
        trapezium,
        trapezium left angle=76,
        trapezium right angle=104,
        trapezium stretches=true,
        box,
        fill=cataGreen,
        draw=cataGreenBorder,
        minimum width=2.20cm
    },
    diamondbox/.style={
        diamond,
        aspect=1.72,
        line width=1.0pt,
        align=center,
        font=\sffamily\small,
        inner sep=2pt,
        fill=cataVisual,
        draw=cataVisualBorder,
        minimum width=1.70cm,
        minimum height=1.45cm
    },
    hexbox/.style={
        regular polygon,
        regular polygon sides=6,
        shape border rotate=30,
        line width=1.0pt,
        align=center,
        font=\sffamily\small,
        inner sep=2pt,
        fill=cataTda,
        draw=cataTdaBorder,
        minimum width=1.90cm,
        minimum height=1.12cm
    },
    qwenbox/.style={box, fill=cataQwen, draw=cataQwenBorder, minimum width=2.02cm, minimum height=1.35cm},
    adapterbox/.style={box, fill=cataAdapter, draw=cataAdapterBorder, minimum width=2.20cm, minimum height=1.18cm},
    evidencebox/.style={box, fill=cataEvidence, draw=cataEvidenceBorder, minimum width=2.45cm, minimum height=0.82cm},
    questionbox/.style={box, fill=cataVit, draw=cataLine, minimum width=1.18cm, minimum height=0.86cm},
    taskbox/.style={box, fill=cataTask, draw=cataTaskBorder, minimum width=2.28cm, minimum height=0.88cm},
    groupbox/.style={rounded corners=8pt, draw=cataTeal, dashed, line width=1.0pt, fill=white},
    title/.style={font=\sffamily\small\bfseries, text=cataLine},
    tinylabel/.style={font=\sffamily\scriptsize, text=cataLine!82}
]

\begin{pgfonlayer}{background}
    \filldraw[groupbox] (0.25,-5.25) rectangle (10.75,0.82);
    \filldraw[groupbox] (11.10,-5.25) rectangle (16.35,2.92);
\end{pgfonlayer}
\node[title] at (5.50,1.10) {Vision + Topology stream};
\node[title] at (13.72,3.20) {Language generation stream};

\node[draw=cataLine, rounded corners=2pt, line width=0.95pt, minimum width=0.58cm, minimum height=0.58cm, inner sep=0pt] (imgicon) at (1.15,-1.35) {};
\draw[cataLine, line width=0.75pt] ($(imgicon.south west)+(0.08,0.12)$) -- ($(imgicon.center)+(-0.06,-0.05)$) -- ($(imgicon.south east)+(-0.07,0.18)$);
\draw[cataLine, line width=0.75pt] ($(imgicon.north west)+(0.16,-0.13)$) circle (0.055cm);
\node[tinylabel, align=center] at (1.15,-2.02) {Endoscopy Image};

\node[vitbox] (vit) at (3.05,-1.35) {\textbf{Frozen ViT}\\[-1pt]\scriptsize Image Encoder};
\node[diamondbox] (visualtda) at (5.60,-1.35) {\textbf{Visual TDA}\\[-1pt]\scriptsize Fusion};
\node[greenpara] (fused) at (8.55,-1.35) {\textbf{Fused Visual}};

\node[hexbox] (patchtda) at (5.60,-3.38) {\textbf{Patch-level TDA}\\[-1pt]\scriptsize Descriptors};
\node[greenpara] (condition) at (8.55,-3.38) {\textbf{TDA Condition}};
\node[evidencebox] (evidence) at (8.55,-4.72) {\textbf{Explanation}\\[-1pt]\scriptsize Evidence};

\node[questionbox] (question) at (12.62,1.68) {\textbf{Question}};
\node[qwenbox] (qwen) at (12.62,-1.35) {\textbf{Qwen2.5-3B}\\[-1pt]\scriptsize Instruct\\[-1pt]\scriptsize + QLoRA};
\node[adapterbox] (adapters) at (12.62,-3.38) {\textbf{Gated TDA}\\[-1pt]\textbf{Adapter}};

\node[draw=cataLine, rounded corners=2pt, line width=0.95pt, minimum width=0.58cm, minimum height=0.62cm, inner sep=0pt] (answericon) at (14.95,-1.35) {};
\draw[cataLine, line width=0.75pt] ($(answericon.north west)+(0.12,-0.13)$) -- ($(answericon.north east)+(-0.12,-0.13)$);
\draw[cataLine, line width=0.75pt] ($(answericon.west)+(0.12,0.00)$) -- ($(answericon.east)+(-0.16,0.00)$);
\draw[cataLine, line width=0.75pt] ($(answericon.south west)+(0.12,0.13)$) -- ($(answericon.south east)+(-0.25,0.13)$);
\node[tinylabel, align=center] at (14.95,-2.02) {Generated Answer};

\node[taskbox] (task2) at (14.95,-4.72) {\textbf{Task 2: Explanation}\\[-1pt]\scriptsize + Confidence};

\draw[flow] (imgicon.east) -- (vit.west);
\draw[flow] (vit.east) -- (visualtda.west);
\draw[flow] (visualtda.east) -- (fused.west);
\draw[flow] (fused.east) -- (qwen.west);
\draw[flow] (qwen.east) -- (answericon.west);

\draw[flow] (question.south) -- (qwen.north);
\draw[flow] (imgicon.south) -- ++(0,-1.82) -| (patchtda.west);
\draw[flow] (patchtda.north) -- (visualtda.south);
\draw[flow] (patchtda.east) -- (condition.west);
\draw[flow] (condition.east) -- (adapters.west);
\draw[flow] (condition.south) -- (evidence.north);
\draw[flow] (adapters.north) -- (qwen.south);
\draw[flow] (adapters.south) -- ++(0,-1.02) -| (evidence.east);
\draw[flow] (evidence.east) -- (task2.west);

\end{tikzpicture}
\end{document}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the fixed-coordinate CATA architecture figure.")
    parser.add_argument(
        "--plotneuralnet",
        default="",
        help="Ignored. Kept for backward compatibility with the old command.",
    )
    parser.add_argument(
        "--output",
        default="paper_assets/figures/cata_architecture",
        help="Output path without extension or with .tex extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.suffix.lower() != ".tex":
        output = output.with_suffix(".tex")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(CATA_ARCH_TEX, encoding="utf-8")
    print(f"Wrote CATA architecture TeX to: {output}")


if __name__ == "__main__":
    main()
