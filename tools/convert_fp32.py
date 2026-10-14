"""Convertit le modèle core fp16 en fp32 complet (poids, constantes, Cast).

Usage : python tools/convert_fp32.py entree_fp16.onnx sortie_fp32.onnx

Pourquoi : le graphe publié calcule en demi-précision ; sur carte graphique
(DirectML) la variance du prologue déborde (> 65504) et toute la sortie
devient NaN. En fp32 le calcul est exact partout.
"""
from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def _to_fp32(t: TensorProto) -> TensorProto:
    if t.data_type != TensorProto.FLOAT16:
        return t
    arr = numpy_helper.to_array(t).astype(np.float32)
    new = numpy_helper.from_array(arr, t.name)
    return new


def convert(src: str, dst: str) -> None:
    m = onnx.load(src)
    g = m.graph
    n_init = n_cast = n_const = n_vi = 0
    for i, t in enumerate(g.initializer):
        if t.data_type == TensorProto.FLOAT16:
            g.initializer[i].CopyFrom(_to_fp32(t))
            n_init += 1
    for n in g.node:
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == TensorProto.FLOAT16:
                    a.i = TensorProto.FLOAT
                    n_cast += 1
        elif n.op_type in ("Constant", "ConstantOfShape"):
            for a in n.attribute:
                if a.name == "value" and a.t.data_type == TensorProto.FLOAT16:
                    a.t.CopyFrom(_to_fp32(a.t))
                    n_const += 1
    for coll in (g.value_info, g.input, g.output):
        for v in coll:
            tt = v.type.tensor_type
            if tt.elem_type == TensorProto.FLOAT16:
                tt.elem_type = TensorProto.FLOAT
                n_vi += 1
    # pas de check_model : le graphe source n'est pas trié topologiquement (ORT s'en charge)
    onnx.save(m, dst)
    print(f"converti : {n_init} poids, {n_cast} Cast, {n_const} constantes, {n_vi} types -> {dst}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
